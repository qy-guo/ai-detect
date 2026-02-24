import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5' # GPU
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from utils import generate_prompt, extract_answer

# --- 1. 定义路径 ---
# 基础模型路径
BASE_MODEL_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LLM/models/qwen1.5-7b-Chat"

# LoRA适配器路径 (final_model 目录)
LORA_ADAPTER_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/output/qlora_3/final_model"
LORA_ADAPTER_NAME = "qlora_adapter"
EVAL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(LORA_ADAPTER_PATH)), "evaluation_results")

# 测试集路径
TEST_DATA_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean/test.jsonl"
NEW_TEST_DATA_GPT5_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean/new_test_gpt5_origin.jsonl"
NEW_TEST_DATA_GPT5_LIKE_HUMAN_MID_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean/new_test_gpt5_like_human_mid.jsonl"
NEW_TEST_DATA_GPT5_LIKE_HUMAN_STRONG_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean/new_test_gpt5_like_human_strong.jsonl"


# --- 2. 加载vLLM模型和分词器 ---
print("正在加载vLLM引擎和LoRA适配器...")

# 使用所有8张GPU进行张量并行
llm = LLM(
    model=BASE_MODEL_PATH,
    tensor_parallel_size=4,
    enable_lora=True,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=1024  # 限制KV缓存，防止OOM
)
print("vLLM引擎加载完毕。")

print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("分词器加载完毕。")


# --- 3. 定义辅助函数和评估流程 ---
def safe_load_jsonl(path):
    """安全地从jsonl文件加载数据，忽略空行。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 测试文件未找到于路径 {path}")
        return None

def save_classified_samples(output_dir_base, test_set_name, test_data, ground_truths, predictions):
    """根据预测结果将样本分类为TP, FP, TN, FN并保存。"""
    
    # 1. 创建特定于本次测试的输出目录
    test_set_name_sanitized = "".join(c for c in test_set_name if c.isalnum() or c in (' ', '_', '.')).rstrip().replace(' ', '_')
    eval_output_dir = os.path.join(output_dir_base, test_set_name_sanitized)
    os.makedirs(eval_output_dir, exist_ok=True)
    print(f"分类结果将保存至: {eval_output_dir}")

    # 2. 准备分类列表
    tp_samples, fp_samples, tn_samples, fn_samples = [], [], [], []

    # 3. 遍历所有样本进行分类
    for i in range(len(test_data)):
        prediction = predictions[i]
        ground_truth = ground_truths[i]
        
        if prediction not in [0, 1]:
            continue

        sample_with_pred = test_data[i].copy()
        sample_with_pred['predicted_source'] = "ai" if prediction == 1 else "human"
        
        if ground_truth == 1 and prediction == 1:  # TP: AI correctly identified as AI
            tp_samples.append(sample_with_pred)
        elif ground_truth == 0 and prediction == 1: # FP: Human incorrectly identified as AI
            fp_samples.append(sample_with_pred)
        elif ground_truth == 0 and prediction == 0: # TN: Human correctly identified as Human
            tn_samples.append(sample_with_pred)
        elif ground_truth == 1 and prediction == 0: # FN: AI incorrectly identified as Human
            fn_samples.append(sample_with_pred)

    # 4. 定义保存函数并保存文件
    def _save_to_jsonl(data, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  - 已保存 {len(data):>4} 个样本到 {os.path.basename(filepath)}")

    _save_to_jsonl(tp_samples, os.path.join(eval_output_dir, "TP_AI_is_AI.jsonl"))
    _save_to_jsonl(tn_samples, os.path.join(eval_output_dir, "TN_Human_is_Human.jsonl"))
    _save_to_jsonl(fp_samples, os.path.join(eval_output_dir, "FP_Human_is_AI.jsonl"))
    _save_to_jsonl(fn_samples, os.path.join(eval_output_dir, "FN_AI_is_Human.jsonl"))

def evaluate_on_test_set(test_data_path, test_set_name, llm, tokenizer):
    """在给定的测试集上使用vLLM执行完整的评估流程。"""
    print(f"\n{'='*20} 开始评估: {test_set_name} {'='*20}")
    
    test_data = safe_load_jsonl(test_data_path)
    if not test_data:
        return

    # --- 准备所有prompts ---
    prompts = []
    ground_truths = []
    system_prompt = "你是一个文本来源分析专家助手，需要严格按照我的指令并结合你的专业知识来回答问题。"

    for item in test_data:
        user_prompt = generate_prompt(item["question"], item["answer"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(formatted_prompt)
        ground_truths.append(0 if item["source"] == "human" else 1)

    # --- vLLM批量推理 ---
    print(f"开始在 {len(prompts)} 条测试数据上进行批量推理...")
    sampling_params = SamplingParams(max_tokens=3, temperature=0)
    
    # vLLM 新版 LoRA API 用法
    lora_request = LoRARequest(LORA_ADAPTER_NAME, 1, LORA_ADAPTER_PATH)
    
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    
    # --- 解析结果 ---
    predictions = []
    for output in tqdm(outputs, desc=f"解析结果 ({test_set_name})"):
        response_text = output.outputs[0].text.strip()
        predicted_label, _ = extract_answer(response_text)
        predictions.append(predicted_label)

    # --- 新增功能：根据预测结果对样本进行分类并保存 ---
    save_classified_samples(EVAL_OUTPUT_DIR, test_set_name, test_data, ground_truths, predictions)

    # --- 计算指标 ---
    valid_indices = [i for i, p in enumerate(predictions) if p in [0, 1]]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]
    valid_predictions = [predictions[i] for i in valid_indices]

    print(f"\n--- {test_set_name} 评估报告 ---")
    if len(valid_predictions) > 0:

        cm = confusion_matrix(valid_ground_truths, valid_predictions, labels=[0, 1])
        # 保证当cm不是2x2时，代码不会出错
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else: # 处理只有一个类别被预测的情况
            if len(valid_predictions) > 0 and valid_predictions[0] == 0:
                 tn, fp, fn, tp = len(valid_predictions), 0, 0, 0
            else:
                 tn, fp, fn, tp = 0, 0, 0, len(valid_predictions)


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
        "新测试集 (GPT-5 Like Human Mid)": NEW_TEST_DATA_GPT5_LIKE_HUMAN_MID_PATH,
        "新测试集 (GPT-5 Like Human Strong)": NEW_TEST_DATA_GPT5_LIKE_HUMAN_STRONG_PATH,
    }


    for name, path in test_sets.items():
        evaluate_on_test_set(path, name, llm, tokenizer)

if __name__ == "__main__":
    main()
