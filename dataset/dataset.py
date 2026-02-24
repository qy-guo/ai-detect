import json
import numpy as np
import os


def safe_load_jsonl(path):
    """
    加载 JSONL 中每个 JSON 文件，
    返回一个包含 JSON 数据的 List
    """
    with open(path, "r", encoding="utf-8") as f:
        data = []
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                line = json.loads(line)
                data.append(line)
            except json.JSONDecodeError as e:
                print(f"第 {idx} 行加载错误: {e}")
    return data


def process_data(dataset):
    """
    处理划分好的训练、验证、测试集，将每条数据转换为 2 个问答对，例如：
    {
        "question": "...",
        "answer": "...",
        "source": "..."     # huamn 或 ai
    }
    """
    processed_dataset = []
    for data in dataset:
        question = data["question"]
        human_answer = data["human_answers"][0]
        ai_answer = data["chatgpt_answers"][0]
        human_data = {
            "question": question,
            "answer": human_answer,
            "source": "human"
        }
        ai_data = {
            "question": question,
            "answer": ai_answer,
            "source": "ai"
        }
        processed_dataset.append(human_data)
        processed_dataset.append(ai_data)
    return processed_dataset

def split_data(data, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """
    随机选择 80% 作为训练集、10% 作为验证集、10% 作为测试集，
    并处理数据集，转换为问答对
    """
    n = len(data)
    a = int(n * train_ratio)
    b = int(n * (train_ratio + valid_ratio))
    # 随机数生成器
    rng = np.random.default_rng(seed)
    # 打乱数组
    data_shuffled = rng.permutation(data)
    # 训练集
    train = data_shuffled[: a].tolist()
    valid = data_shuffled[a: b].tolist()
    test = data_shuffled[b:].tolist()
    return process_data(train), process_data(valid), process_data(test)

def save_to_jsonl(save_path, data):
    with open(save_path, "w", encoding="utf-8") as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + "\n")
    print(f"成功写入 {len(data)} 个样本到 {save_path}")


if __name__ == "__main__":
    # 加载数据集
    # 百科(共4617)
    data_baike_path = r"/media/tgliu/hdd/qyguo_llm/LoRA/dataset/AI-ModelScope/HC3-Chinese/baike.jsonl"
    data_baike = safe_load_jsonl(data_baike_path)
    # 经济(共689)
    data_finance_path = r"/media/tgliu/hdd/qyguo_llm/LoRA/dataset/AI-ModelScope/HC3-Chinese/finance.jsonl"
    data_finance = safe_load_jsonl(data_finance_path)
    # 法律(共372)
    data_law_path = r"/media/tgliu/hdd/qyguo_llm/LoRA/dataset/AI-ModelScope/HC3-Chinese/law.jsonl"
    data_law = safe_load_jsonl(data_law_path)
    # 医学(共1074)
    data_medicine_path = r"/media/tgliu/hdd/qyguo_llm/LoRA/dataset/AI-ModelScope/HC3-Chinese/medicine.jsonl"
    data_medicine = safe_load_jsonl(data_medicine_path)
    # 划分数据集
    train_baike, valid_baike, test_baike = split_data(data_baike)
    train_finance, valid_finance, test_finance = split_data(data_finance)
    train_law, valid_law, test_law = split_data(data_law)
    train_medicine, valid_medicine, test_medicine = split_data(data_medicine)
    
    # 训练
    train_data = train_baike + train_finance + train_law + train_medicine
    # 验证
    valid_data = valid_baike + valid_finance + valid_law + valid_medicine
    # 测试
    test_data = test_baike + test_finance + test_law + test_medicine

    # 保存数据集
    save_path = r"/media/tgliu/hdd/qyguo_llm/LoRA/dataset/processed"
    os.makedirs(save_path, exist_ok=True)
    
    train_path = os.path.join(save_path, "train.jsonl")
    save_to_jsonl(train_path, train_data)
    
    valid_path = os.path.join(save_path, "valid.jsonl")
    save_to_jsonl(valid_path, valid_data)
    
    test_path = os.path.join(save_path, "test.jsonl")
    save_to_jsonl(test_path, test_data)
    
    
    
    
        
    