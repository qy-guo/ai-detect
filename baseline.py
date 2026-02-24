"""
使用未经过微调的 Qwen1.5-7B-Chat 在 HC3-CHinese 测试集上进行推理
"""
from utils import generate_prompt, extract_answer, load_model_and_tokenizer, infer
from dataset import safe_load_jsonl
import logging
import json


# logging
logging.basicConfig(
    filename=r"/media/tgliu/45edc4ec-70ab-4df9-8d6c-1367d9e83f01/qyguo_llm/LoRA/baseline.log",
    filemode="a",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 本地模型路径
MODEL_PATH = r"/media/tgliu/45edc4ec-70ab-4df9-8d6c-1367d9e83f01/qyguo_llm/LLM/models/qwen1.5-7b-Chat"

# 加载测试集
test_path = r"/media/tgliu/45edc4ec-70ab-4df9-8d6c-1367d9e83f01/qyguo_llm/LoRA/dataset/processed/test.jsonl"
test_data = safe_load_jsonl(test_path)
n = len(test_data)

# 加载大模型到显存中
model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

# 依次遍历每个测试样本
outputs = []
num_succeed, num_failed = 0, 0
num_human, num_ai = 0, 0
for idx, line in enumerate(test_data, 1):
    # 提取 Q、A、label
    try:
        question = line["question"]
        answer = line["answer"]
        source = line["source"]
        label = 0 if source == "human" else 1
        # 构建 prompt
        prompt = generate_prompt(question, answer)
        # 推理
        response = infer(model, tokenizer, user_prompt=prompt, max_new_tokens=3)
        # 提取回答
        response_text = extract_answer(response)
        outputs.append(response_text)
        # 记录日志
        if response_text[0] == 0:
            num_human += 1
            logging.info(f"{idx}/{n} 处理成功，判断为：[0]人类")
        elif response_text[0] == 1:
            num_ai += 1
            logging.info(f"{idx}/{n} 处理成功，判断为：[1]AI")
        else:
            logging.info(f"{idx}/{n} 处理成功，判断为：[2]{response_text[1]}")
        num_succeed += 1
    except Exception as e:
        num_failed += 1
        logging.warning(f"{idx}/{n} 处理失败: {e}")

logging.info(f"处理完成，共成功 {num_succeed}/{n} 条，失败 {num_failed}/{n} 条")
logging.info(f"判断为人类回答共 {num_human}/{n} 条，AI回答共 {num_ai}/{n} 条")

output_path = r"/media/tgliu/45edc4ec-70ab-4df9-8d6c-1367d9e83f01/qyguo_llm/LoRA/output/baseline.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(outputs, f, indent=2, ensure_ascii=False)