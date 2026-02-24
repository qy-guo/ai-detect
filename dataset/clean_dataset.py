import os
import json
import logging
import re
from collections import defaultdict

# --- Configuration ---
# 设置日志记录
LOG_DIR = r'/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean'
LOG_FILE = os.path.join(LOG_DIR, 'clean.log')
# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

# 定义数据文件路径
# __file__ 是当前脚本的路径
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed', 'raw')
CLEAN_DIR = r'/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean'   # 清洗后文件的输出目录
FILES_TO_CLEAN = ['new_test_gpt5_like_human_mid.jsonl', 'new_test_gpt5_like_human_strong.jsonl', 'new_test_gpt5_origin.jsonl']

# --- Cleaning Logic ---

def clean_answer_text(answer):
    """
    通过精确替换移除指定的空白字符和换行符来清洗回答文本。
    """
    if not isinstance(answer, str):
        return None
    # 按照从最具体到最普遍的顺序进行替换
    # 移除 "\\n\\n"
    cleaned_text = answer.replace("\\n\\n", "")
    # 移除 "\n\n"
    cleaned_text = answer.replace("\n\n", "")
    # 移除剩余的换行符 "\\n"
    cleaned_text = cleaned_text.replace("\\n", "")
    # 移除剩余的换行符 "\n"
    cleaned_text = cleaned_text.replace("\n", "")
    # 移除所有空格 " "
    cleaned_text = cleaned_text.replace(" ", "")
    
    # [可选] 作为保障，再用一次正则表达式移除所有其他可能的空白字符
    # cleaned_text = re.sub(r'\s+', '', cleaned_text)
    
    return cleaned_text

def clean_dataset_file(input_filepath, output_filepath):
    """
    根据规则清洗单个.jsonl数据集文件，并将结果保存到新路径。
    """
    if not os.path.exists(input_filepath):
        logging.warning(f"输入文件不存在，跳过: {input_filepath}")
        return

    logging.info(f"--- 开始处理文件: {os.path.basename(input_filepath)} ---")

    # 1. 将所有数据读入内存
    with open(input_filepath, 'r', encoding='utf-8') as f:
        original_data = [json.loads(line) for line in f]
    
    original_count = len(original_data)
    logging.info(f"原始数据行数: {original_count}")

    # 2. 识别因回答无效而需要被删除的问题
    questions_to_discard = set()
    for item in original_data:
        answer = item.get('answer')
        cleaned_answer = clean_answer_text(answer)
        if not cleaned_answer:  # 如果回答是None或清洗后变为空字符串
            questions_to_discard.add(item['question'])

    if questions_to_discard:
        logging.info(f"找到 {len(questions_to_discard)} 个问题，其关联的所有问答对（human/ai）都将因回答无效而被删除。")
    else:
        logging.info("没有发现回答为None或空的数据。")

    # 3. 过滤数据并构建清洗后的数据集
    cleaned_data = []
    removal_log = defaultdict(list)

    for item in original_data:
        question = item.get('question')
        if question in questions_to_discard:
            # 记录被删除的条目信息以供日志审查
            removal_log[question].append(item.get('source', 'N/A'))
            continue  # 跳过此条目
        
        # 清洗回答文本
        answer = item.get('answer')
        item['answer'] = clean_answer_text(answer)
        cleaned_data.append(item)
    
    # 记录详细的删除日志
    if questions_to_discard:
        total_removed_items = sum(len(v) for v in removal_log.values())
        logging.info(f"总共删除了 {total_removed_items} 个问答对。")
        # 打印一些被删除的例子
        for i, (q, sources) in enumerate(removal_log.items()):
            if i < 5: # 只打印前5个作为示例
                logging.info(f"  - 问题 '{q[:80]}...' 关联的 {len(sources)} 个问答对 ({', '.join(sources)}) 已被删除。")

    # 4. 将清洗后的数据写入新文件
    logging.info(f"正在将清洗后的数据写入: {output_filepath}")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    final_count = len(cleaned_data)
    logging.info(f"清洗后数据行数: {final_count}")
    logging.info(f"共计移除了 {original_count - final_count} 行数据。")
    logging.info(f"--- 文件处理完成: {os.path.basename(input_filepath)} ---\n")

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("==============================================")
    logging.info("          开始执行数据集清洗脚本")
    logging.info("==============================================")
    
    # 确保输出目录存在
    os.makedirs(CLEAN_DIR, exist_ok=True)
    logging.info(f"清洗后的文件将保存到: {CLEAN_DIR}")
    
    for filename in FILES_TO_CLEAN:
        input_path = os.path.join(PROCESSED_DIR, filename)
        output_path = os.path.join(CLEAN_DIR, filename)
        clean_dataset_file(input_path, output_path)
        
    logging.info("所有文件清洗完毕！")
    logging.info(f"详细日志已保存至: {LOG_FILE}")
