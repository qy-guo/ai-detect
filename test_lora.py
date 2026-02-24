import os
import torch
import json
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from utils import generate_prompt, extract_answer # 复用我们的工具函数

# --- 1. 定义路径 ---
# 基础模型路径 (必须与训练时使用的模型一致)
BASE_MODEL_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LLM/models/qwen1.5-7b-Chat"

# LoRA适配器路径 (训练后保存的路径)
LORA_ADAPTER_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/output/lora_2/final_model"

# 测试集路径
TEST_DATA_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean/test.jsonl"
NEW_TEST_DATA_GPT5_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean/new_test_gpt5_origin.jsonl"
NEW_TEST_DATA_GPT5_LIKE_HUMAN_MID_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean/new_test_gpt5_like_human_mid.jsonl"
NEW_TEST_DATA_GPT5_LIKE_HUMAN_STRONG_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean/new_test_gpt5_like_human_strong.jsonl"

# --- 2. 加载模型和LoRA适配器 ---
print("正在加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"正在将LoRA适配器从 {LORA_ADAPTER_PATH} 应用到基础模型上...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
# 可选：如果后续不再进行训练，可以合并权重以提高推理速度
# model = model.merge_and_unload() 
print("模型加载并合并LoRA适配器完毕！")


# --- 3. 定义辅助函数和评估流程 ---
def safe_load_jsonl(path):
    """安全地从jsonl文件加载数据，忽略空行。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 测试文件未找到于路径 {path}")
        return None

def infer_with_lora(model, tokenizer, user_prompt):
    """使用加载的LoRA模型进行推理。"""
    system_prompt="你是一个文本来源分析专家助手，需要严格按照我的指令并结合你的专业知识来回答问题。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def evaluate_on_test_set(test_data_path, test_set_name, model, tokenizer):
    """在给定的测试集上执行完整的评估流程。"""
    print(f"\n{'='*20} 开始评估: {test_set_name} {'='*20}")
    
    test_data = safe_load_jsonl(test_data_path)
    if test_data is None:
        return

    ground_truths = []
    predictions = []

    print(f"开始在 {len(test_data)} 条测试数据上进行推理...")
    for item in tqdm(test_data, desc=f"推理 ({test_set_name})"):
        try:
            prompt = generate_prompt(item["question"], item["answer"])
            label = 0 if item["source"] == "human" else 1
            ground_truths.append(label)

            response_text = infer_with_lora(model, tokenizer, prompt)
            
            predicted_label, _ = extract_answer(response_text)
            predictions.append(predicted_label)
        
        except Exception as e:
            print(f"处理样本时发生错误: {e}")
            predictions.append(-1) # 标记为错误

    valid_indices = [i for i, p in enumerate(predictions) if p in [0, 1]]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]
    valid_predictions = [predictions[i] for i in valid_indices]

    print(f"\n--- {test_set_name} 评估报告 ---")
    if len(valid_predictions) > 0:

        # --- 混淆矩阵与各项指标 ---
        cm = confusion_matrix(valid_ground_truths, valid_predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0           # 召回率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0      # 特异度
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0        # 精确率
        mcc = matthews_corrcoef(valid_ground_truths, valid_predictions)

        print("混淆矩阵 (Confusion Matrix):")
        print(cm)
        print("\n详细指标:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - Specificity: {specificity:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - MCC: {mcc:.4f}")
        
        # --- 样本统计 ---
        total_samples = len(test_data)
        valid_samples = len(valid_predictions)
        unparsed_samples = predictions.count(2)
        error_samples = predictions.count(-1)
        
        print("\n样本统计:")
        print(f"  - 总样本数          : {total_samples}")
        print(f"  - 有效评估样本数    : {valid_samples} ({valid_samples/total_samples:.2%})")
        print(f"  - 无法解析的回答数: {unparsed_samples} ({unparsed_samples/total_samples:.2%})")
        print(f"  - 处理失败的样本数  : {error_samples} ({error_samples/total_samples:.2%})")
    else:
        print("没有有效的预测结果可以用于评估。")
    print(f"{'='*20} 评估结束: {test_set_name} {'='*20}")

# --- 4. 主执行函数 ---
def main():
    """主函数，按顺序执行所有测试集评估。"""
    test_sets = {
        "原始测试集 (test.jsonl)": TEST_DATA_PATH,
        "新测试集 (GPT-5 Origin)": NEW_TEST_DATA_GPT5_PATH,
        "新测试集 (GPT-5 Like Human)": NEW_TEST_DATA_GPT5_LIKE_HUMAN_MID_PATH,
        "新测试集 (GPT-5 Like Human Strong)": NEW_TEST_DATA_GPT5_LIKE_HUMAN_STRONG_PATH,
    }

    for name, path in test_sets.items():
        evaluate_on_test_set(path, name, model, tokenizer)

if __name__ == "__main__":
    main()
