import json
from dataset import safe_load_jsonl
import re
import os
from tqdm import tqdm
import sys
sys.path.append(r"/storage/public/liutg/qyguo/qyguo_llm/LoRA")
from utils import query, cal_cost
from concurrent.futures import ThreadPoolExecutor, as_completed


def query_worker(gpt, msg, data_human, data_ai, response_type, idx):
    """
    包装 query 函数，使其可以在多线程中携带额外信息
    """
    response = query(gpt, msg)
    return response, data_human, data_ai, response_type, idx


def save_to_jsonl(save_path, data):
    with open(save_path, "w", encoding="utf-8") as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + "\n")
    print(f"成功写入 {len(data)} 个样本到 {save_path}")

def count_answer(answer):
    """
    统计 answer 中的汉字数、句子数
    """
    # 1) 字数
    char_count = len(answer.strip())

    # 2) 句子数：按句末标点计数，末尾引号/括号不影响
    # 终止符：中文全角的 “。！？…”，以及 ASCII 的 !?；多个连续视为一次
    # 允许在终止符后跟若干右引号/右括号类字符
    terminator_pattern = r'[。！？!?…]+(?:[”’」』）】》〉\)\]\}]+)?'
    sentence_count = len(re.findall(terminator_pattern, answer))

    # 若没有终止符但文本中含有至少一个汉字，按 1 句处理（如标题/小红书式短句）
    if sentence_count == 0 and char_count > 0:
        sentence_count = 1

    return  (
        char_count if char_count > 0 else 20, 
        sentence_count
        )

def gpt5_prompt_origin(question, num_word, num_sentense, tolerance=5):
    """
    构建用于生成回答的 prompt
    """
    system_prompt = """
    你是中文技术写作者，仅输出最终答案本身。
    不用列表、标题、代码块或Markdown装饰。
    不用总结提问，避免回答问题的答案之外的任何内容。
    默认用简体中文，语气自然克制，信息准确。
    """
    prompt = f"""
    请直接回答问题，不引入外部资料与新事实。
    【限制】
    - 将答案控制在约 {num_word} 字，允许上下浮动 {tolerance} 字。
    - 控制回答的句子数为 {num_sentense} 句，段落数为 1。
    - 若术语多义，在常见语境下给出主流含义。
    - 在输出前自查是否满足长度/句子/段落/禁语要求；不满足则自行改写到满足后再输出。不要展示自检过程。

    【问题】
    {question}
    """
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return message
    

def gpt5_prompt_like_human_mid(question, num_word, num_sentense, tolerance=5):
    """
    构建用于生成回答的 prompt，不刻意模仿人类语气
    """
    system_prompt = """
    你是中文技术写作者，仅输出最终答案本身。
    不要暴露AI身份；不要说“感谢你的提问/作为AI/抱歉/我不能/无法上网/我不是……”等口癖。
    不用列表、标题、代码块或Markdown装饰。
    不用总结提问，避免回答问题的答案之外的任何内容。
    默认用简体中文，语气自然克制，信息准确，不展示任何思维过程或自检过程。
    """
    prompt = f"""
    请直接回答问题，不引入外部资料与新事实。
    【限制】
    - 将答案控制在约 {num_word} 字，允许上下浮动 {tolerance} 字。
    - 控制回答的句子数为 {num_sentense} 句，段落数为 1。
    - 回答的风格自然叙述，避免模板化开头（如“首先/其次/最后/简单来说”），避免口头禅。
    - 不得出现下列短语：感谢你的提问、作为AI、作为一个AI、抱歉、我不能、我无法上网、我不是……
    - 若术语多义，在常见语境下给出主流含义。
    - 在输出前自查是否满足长度/句子/段落/禁语要求；不满足则自行改写到满足后再输出。不要展示自检过程。

    【问题】
    {question}
    """
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return message

def gpt5_prompt_like_human_strong(question, num_word, num_sentense, tolerance=5):
    """
    构建用于生成回答的 prompt，刻意模仿人类语气
    """
    system_prompt = """
    你是中文技术写作者，仅输出最终答案本身。
    不要暴露AI身份；不要说“感谢你的提问/作为AI/抱歉/我不能/无法上网/我不是……”等口癖。
    不用列表、标题、代码块或Markdown装饰。
    不用总结提问，避免回答问题的答案之外的任何内容。
    默认用简体中文，语气自然克制，信息准确，不展示任何思维过程或自检过程。
    """
    prompt = f"""
    把下面问题用更“人类化”的方式回答：自然、有节奏，但不引入新事实或外部资料。
    【限制】
    - 尽可能地模仿真实人类的回答口吻；
    - 将答案控制在约 {num_word} 字，允许上下浮动 {tolerance} 字。
    - 控制回答的句子数为 {num_sentense} 句，段落数为 1。
    - 回答的风格自然叙述，避免模板化开头（如“首先/其次/最后/简单来说”），避免口头禅。
    - 不得出现下列短语：感谢你的提问、作为AI、作为一个AI、抱歉、我不能、我无法上网、我不是……
    - 若术语多义，在常见语境下给出主流含义。
    - 在输出前自查是否满足长度/句子/段落/禁语要求；不满足则自行改写到满足后再输出。不要展示自检过程。

    【问题】
    {question}
    """
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return message

def generate_test_data_by_gpt5(testdata_path, gpt="gpt-5-chat-latest", total_cost={}):
    """
    针对测试集每个 question，分别生成：
    1. 无指定口吻的 GPT5 回答
    2. 指定要求模仿人类口吻的 GPT5 回答
    且要求生成的回答字数、句子数、段落数与人类回答统一
    """
    # 数据集
    test_gpt5_origin = []
    test_gpt5_like_human_mid = []
    test_gpt5_like_human_strong = []
    # 读取 testdata，返回一个 List
    testdata = safe_load_jsonl(testdata_path)
    # 逐个遍历每个 question
    for idx in tqdm(range(0, len(testdata), 2), desc="处理测试集样本"):
        # 人类问答对
        data_human = testdata[idx]
        answer_human = data_human.get("answer")
        # AI 问答对
        data_ai = testdata[idx+1]
        question_ai = data_ai.get("question")
        # 获取人类回答的特征
        num_word, num_sentence = count_answer(answer_human)
        # 原始 GPT5 回答
        msg_origin = gpt5_prompt_origin(question_ai, num_word, num_sentence)
        response_origin = query(gpt, msg_origin)
        cal_cost(response_json=response_origin.model_dump(), model_name=gpt, stage_name=None, total_accumulated_cost=total_cost)
        response_origin_text = response_origin.choices[0].message.content
        # 构建新测试集
        test_gpt5_origin.append(data_human)
        test_gpt5_origin.append(
            {
                "question": question_ai,
                "answer": response_origin_text.strip(),
                "source": "ai"
            }
        )
        
        # 中等指定口吻的 GPT5 回答
        msg_like_human_mid = gpt5_prompt_like_human_mid(question_ai, num_word, num_sentence)
        response_like_human_mid = query(gpt, msg_like_human_mid)
        cal_cost(response_json=response_like_human_mid.model_dump(), model_name=gpt, stage_name=None, total_accumulated_cost=total_cost)
        response_like_human_mid_text = response_like_human_mid.choices[0].message.content
        # 构建新测试集
        test_gpt5_like_human_mid.append(data_human)
        test_gpt5_like_human_mid.append(
            {
                "question": question_ai,
                "answer": response_like_human_mid_text.strip(),
                "source": "ai"
            }
        )
        
        # 强烈指定要求模仿人类口吻的 GPT5 回答
        msg_like_human_strong = gpt5_prompt_like_human_strong(question_ai, num_word, num_sentence)
        response_like_human_strong = query(gpt, msg_like_human_strong)
        cal_cost(response_json=response_like_human_strong.model_dump(), model_name=gpt, stage_name=None, total_accumulated_cost=total_cost)
        response_like_human_strong_text = response_like_human_strong.choices[0].message.content
        # 构建新测试集
        test_gpt5_like_human_strong.append(data_human)
        test_gpt5_like_human_strong.append(
            {
                "question": question_ai,
                "answer": response_like_human_strong_text.strip(),
                "source": "ai"
            }
        )
    
    return (test_gpt5_origin, test_gpt5_like_human_mid, test_gpt5_like_human_strong, total_cost)

def generate_test_data_by_gpt5_concurrent(testdata_path, gpt="gpt-5-chat-latest", total_cost={}, max_workers=10):
    """
    针对测试集每个 question，并发生成（保证顺序）：
    - origin：原始 GPT5 回答
    - like_human_mid：中等指定模仿人类口吻的 GPT5 回答
    - like_human_strong：强烈要求模仿人类口吻的 GPT5 回答
    且要求生成的回答字数、句子数与人类回答统一
    """
    testdata = safe_load_jsonl(testdata_path)
    
    # 预先初始化结果列表以保证顺序
    test_gpt5_origin = [None] * len(testdata)
    test_gpt5_like_human_mid = [None] * len(testdata)
    test_gpt5_like_human_strong = [None] * len(testdata)
    
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx in range(0, len(testdata), 2):
            # 人类问答对
            data_human = testdata[idx]
            data_ai = testdata[idx+1]
            # AI 问答对
            question_ai = data_ai.get("question")
            answer_human = data_human.get("answer")

            # 跳过人类回答为空的样本
            if not answer_human or not answer_human.strip():
                print(f"警告: 索引 {idx} 的人类回答为空，跳过该样本对。")
                continue

            # 统计人类回答的字数、句子数
            try:
                num_word, num_sentence = count_answer(answer_human)
            except Exception as e:
                print(f"第 {idx} 个问题提取人类回答信息失败")
            
            # 提交 “origin” 任务
            msg_origin = gpt5_prompt_origin(question_ai, num_word, num_sentence)
            tasks.append(executor.submit(query_worker, gpt, msg_origin, data_human, data_ai, "origin", idx))
            
            # 提交 “like_human_mid” 任务
            msg_like_human_mid = gpt5_prompt_like_human_mid(question_ai, num_word, num_sentence)
            tasks.append(executor.submit(query_worker, gpt, msg_like_human_mid, data_human, data_ai, "like_human_mid", idx))
            
            # 提交 “like_human” 任务
            msg_like_human_strong = gpt5_prompt_like_human_strong(question_ai, num_word, num_sentence)
            tasks.append(executor.submit(query_worker, gpt, msg_like_human_strong, data_human, data_ai, "like_human_strong", idx))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="处理测试集样本"):
            try:
                response, data_human, data_ai, response_type, original_idx = future.result()
                
                # 增加健壮性：检查API返回内容是否有效
                if response is None or not response.choices or response.choices[0].message.content is None:
                    print(f"警告: 索引 {original_idx} ({response_type}) 的任务返回内容为空，已跳过。")
                    continue

                cal_cost(response_json=response.model_dump(), model_name=gpt, stage_name=None, total_accumulated_cost=total_cost)
                response_text = response.choices[0].message.content.strip()
                
                new_ai_data = {
                    "question": data_ai.get("question"),
                    "answer": response_text,
                    "source": "ai"
                }
                if response_type == "origin":
                    test_gpt5_origin[original_idx] = data_human
                    test_gpt5_origin[original_idx+1] = new_ai_data
                elif response_type == "like_human_mid":
                    test_gpt5_like_human_mid[original_idx] = data_human
                    test_gpt5_like_human_mid[original_idx + 1] = new_ai_data
                elif response_type == "like_human_strong":
                    test_gpt5_like_human_strong[original_idx] = data_human
                    test_gpt5_like_human_strong[original_idx + 1] = new_ai_data

            except Exception as e:
                print(f"任务执行出错: {e}")

    return (test_gpt5_origin, test_gpt5_like_human_mid, test_gpt5_like_human_strong, total_cost)

if __name__ == "__main__":
    # 测试集路径
    testdata_path = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean/test.jsonl"
    # gpt
    gpt = "gpt-5-chat-latest"
    # 统计 token
    total_cost = {}
    
    # 构建新测试集，保证回答字数、句子数与人类的回答统一
    print("构建新测试集")
    # 1. 原始 GPT 回答
    # 2.中等指定模仿人类口吻的 GPT5 回答
    # 3.强烈要求模仿人类口吻的 GPT5 回答
    # test_gpt5_origin, test_gpt5_like_human_mid, test_gpt5_like_human_strong, total_cost = generate_test_data_by_gpt5(testdata_path, gpt, total_cost=total_cost)
    test_gpt5_origin, test_gpt5_like_human_mid, test_gpt5_like_human_strong, total_cost = generate_test_data_by_gpt5_concurrent(testdata_path, gpt, total_cost=total_cost, max_workers=8)
    
    # 在保存前过滤掉所有值为 None 的条目，保证数据完整性
    test_gpt5_origin_clean = [item for item in test_gpt5_origin if item is not None]
    test_gpt5_like_human_mid_clean = [item for item in test_gpt5_like_human_mid if item is not None]
    test_gpt5_like_human_strong_clean = [item for item in test_gpt5_like_human_strong if item is not None]
    
    # 保存新测试集
    test_dir = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/raw"
    test_gpt5_origin_path = os.path.join(test_dir, "new_test_gpt5_origin.jsonl")
    test_gpt5_like_human_mid_path = os.path.join(test_dir, "new_test_gpt5_like_human_mid.jsonl")
    test_gpt5_like_human_strong_path = os.path.join(test_dir, "new_test_gpt5_like_human_strong.jsonl")
    save_to_jsonl(test_gpt5_origin_path, test_gpt5_origin_clean)
    save_to_jsonl(test_gpt5_like_human_mid_path, test_gpt5_like_human_mid_clean)
    save_to_jsonl(test_gpt5_like_human_strong_path, test_gpt5_like_human_strong_clean)
    
    # 消耗 token 情况
    print(json.dumps(total_cost, indent=2, ensure_ascii=False))