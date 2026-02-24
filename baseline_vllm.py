"""
使用未经过微调的 Qwen1.5-7B-Chat 在多个测试集上进行 vLLM 批量推理，并评估性能指标。
"""
import os
import json
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from utils import generate_prompt, extract_answer

# --- 1. 路径定义 ---
BASE_DIR = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA"
MODEL_PATH = r"/storage/public/liutg/qyguo/qyguo_llm/LLM/models/qwen1.5-7b-Chat"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs/baseline_vllm.log")

# 定义所有测试集路径
TEST_SETS = {
    "原始测试集 (test.jsonl)": os.path.join(BASE_DIR, "dataset/processed/clean/test.jsonl"),
    "新测试集 (GPT-5 Origin)": os.path.join(BASE_DIR, "dataset/processed/clean/new_test_gpt5_origin.jsonl"),
    "新测试集 (GPT-5 Like Human Mid)": os.path.join(BASE_DIR, "dataset/processed/clean/new_test_gpt5_like_human_mid.jsonl"),
    "新测试集 (GPT-5 Like Human Strong)": os.path.join(BASE_DIR, "dataset/processed/clean/new_test_gpt5_like_human_strong.jsonl"),
}

# 创建输出和日志目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# --- 2. 日志配置 ---
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w", # 每次重新运行都覆盖旧日志
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- 3. 评估函数 ---
def safe_load_jsonl(path):
    """安全地从jsonl文件加载数据，忽略空行。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 测试文件未找到于路径 {path}")
        return None
    
def evaluate_on_test_set(test_data_path, test_set_name, llm, tokenizer):
    """在给定的测试集上使用vLLM执行完整的评估流程并打印指标。"""
    print(f"\n{'='*20} 开始评估: {test_set_name} {'='*20}")
    logging.info(f"{'='*20} 开始评估: {test_set_name} {'='*20}")
    
    test_data = safe_load_jsonl(test_data_path)
    if not test_data:
        error_msg = f"测试集加载失败或为空，路径: {test_data_path}"
        print(error_msg)
        logging.error(error_msg)
        return

    n = len(test_data)
    logging.info(f"测试集加载成功，共 {n} 条样本。")

    # --- 批量准备推理数据 ---
    prompts = []
    ground_truths = []
    for item in test_data:
        prompt = generate_prompt(item["question"], item["answer"])
        # 在vLLM中，聊天模板需要手动应用
        messages = [{"role": "system", "content": "你是一个文本来源分析专家助手，需要严格按照我的指令并结合你的专业知识来回答问题。"}, {"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(formatted_prompt)
        ground_truths.append(0 if item["source"] == "human" else 1)

    # --- 执行vLLM批量推理 ---
    print(f"开始在 {n} 条测试数据上进行批量推理...")
    sampling_params = SamplingParams(max_tokens=3, temperature=0.0)
    vllm_outputs = llm.generate(prompts, sampling_params)
    print("批量推理完成。")

    # --- 处理并解析结果 ---
    predictions = []
    for output in tqdm(vllm_outputs, desc=f"解析结果 ({test_set_name})"):
        response_text = output.outputs[0].text.strip()
        predicted_label, _ = extract_answer(response_text)
        predictions.append(predicted_label)
        
    # --- 计算并打印评估指标 ---
    valid_indices = [i for i, p in enumerate(predictions) if p in [0, 1]]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]
    valid_predictions = [predictions[i] for i in valid_indices]

    print(f"\n--- {test_set_name} 评估报告 ---")
    logging.info(f"--- {test_set_name} 评估报告 ---")

    if len(valid_predictions) > 0:

        cm = confusion_matrix(valid_ground_truths, valid_predictions, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0 # 简化处理，避免在单类别预测时出错

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0           # 召回率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0      # 特异度
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0        # 精确率
        mcc = matthews_corrcoef(valid_ground_truths, valid_predictions)

        print("混淆矩阵 (Confusion Matrix):")
        print(cm)
        logging.info("混淆矩阵 (Confusion Matrix):\n" + str(cm))
        
        metrics_summary = (
            f"\n详细指标:\n"
            f"  - Accuracy: {accuracy:.4f}\n"
            f"  - Recall: {recall:.4f}\n"
            f"  - Specificity: {specificity:.4f}\n"
            f"  - Precision: {precision:.4f}\n"
            f"  - MCC: {mcc:.4f}"
        )
        print(metrics_summary)
        logging.info(metrics_summary)
        
        total_samples = n
        valid_samples = len(valid_predictions)
        unparsed_samples = predictions.count(2)
        error_samples = predictions.count(-1)
        
        stats_summary = (
            f"\n样本统计:\n"
            f"  - 总样本数          : {total_samples}\n"
            f"  - 有效评估样本数    : {valid_samples} ({valid_samples/total_samples:.2%})\n"
            f"  - 无法解析的回答数: {unparsed_samples} ({unparsed_samples/total_samples:.2%})\n"
            f"  - 处理失败的样本数  : {error_samples} ({error_samples/total_samples:.2%})"
        )
        print(stats_summary)
        logging.info(stats_summary)
    else:
        print("没有有效的预测结果可以用于评估。")
        logging.warning("没有有效的预测结果可以用于评估。")

# --- 4. 主执行函数 ---
def main():
    """主函数，加载模型并按顺序执行所有测试集评估。"""
    logging.info("="*30 + " 开始基线模型vLLM推理任务 " + "="*30)
    
    print("正在加载vLLM引擎...")
    # 通过设置 max_model_len 来限制KV缓存大小，避免OOM
    llm = LLM(
        model=MODEL_PATH, 
        trust_remote_code=True, 
        dtype="bfloat16",
        max_model_len=1024
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("vLLM引擎和分词器加载完毕。")
    logging.info("vLLM引擎和分词器加载完毕。")

    for name, path in TEST_SETS.items():
        evaluate_on_test_set(path, name, llm, tokenizer)
        
    logging.info("="*30 + " 基线模型vLLM推理任务结束 " + "="*30 + "\n")
    print("\n所有评估任务完成。")

if __name__ == "__main__":
    main()
