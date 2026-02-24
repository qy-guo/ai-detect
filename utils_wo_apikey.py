from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import os
import openai
from openai import OpenAI
import httpx
import time


def _query_once(gpt_version, msg, temperature):
    """
    封装后的 API 单词调用函数
    """
    if gpt_version=='deepseek-ai/DeepSeek-V3':
        client = OpenAI(
            api_key="",
            base_url=""
            )
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=msg,
            temperature=temperature,
        )
        
    elif gpt_version=='deepseek-ai/DeepSeek-R1':
        client = OpenAI(
            api_key="",
            base_url=""
            )
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=msg,
            temperature=temperature,
        )
    
    elif gpt_version == "gpt-5-chat-latest":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=msg,
            temperature=temperature,
        )
        
    elif gpt_version == "chatgpt-4o":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="chatgpt-4o",
            messages=msg,
            temperature=temperature,
        )
    
    elif gpt_version == "gemini-2.0-flash":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=msg,
            temperature=temperature,
        )
        
    elif gpt_version == "gemini-1.5-pro":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="gemini-1.5-pro",
            messages=msg,
            temperature=temperature,
        )
    
    elif gpt_version == "gemini-1.5-flash":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=msg,
            temperature=temperature,
        )
    
    elif gpt_version == "gemini-2.5-pro-preview-05-06":
        client = OpenAI(
            api_key="",
            base_url="",
        )
        completion = client.chat.completions.create(
            model="",
            messages=msg,
            temperature=temperature,
        )
    
    elif gpt_version == "claude-sonnet-4-20250514":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="",
            messages=msg,
            temperature=temperature,
            max_tokens=32000,
        )
        
    elif gpt_version == "claude-opus-4-20250514":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="claude-opus-4-20250514",
            messages=msg,
            temperature=temperature,
            max_tokens=32000,
        )
        
    elif gpt_version == "claude-3-7-sonnet-20250219":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="claude-3-7-sonnet-20250219",
            messages=msg,
            temperature=temperature,
            max_tokens=32000,
        )
        
    elif gpt_version == "qwen-plus":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=msg,
            temperature=temperature
        )
    
    elif gpt_version == "qwen-plus-thinking":
        client = OpenAI(
            api_key="",
            base_url=""
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=msg,
            temperature=temperature,
            extra_body={"enable_thinking": True}   # 开启 thinking
        )
    
    return completion 

def query(gpt_version, msg, temperature=1, max_retries=3, backoff=5, logger=None):
    """
    带 logger、重试机制的 API 调用函数
    """
    for attempt in range(1, max_retries + 1):
        try:
            return _query_once(gpt_version, msg, temperature)
        except (openai.APIError, openai.InternalServerError, httpx.HTTPError, Exception) as e:
            info_msg = f"[重试机制] 第 {attempt}/{max_retries} 次调用失败：{type(e).__name__}: {e}"
            print(info_msg)
            if logger:
                logger.info(info_msg)

            if attempt < max_retries:
                info_msg = f"[重试机制] 等待 {backoff} 秒后重试..."
                print(info_msg)
                if logger:
                    logger.info(info_msg)
                time.sleep(backoff)
            else:
                info_msg = f"[重试机制] 超过最大重试次数，抛出异常"
                print(info_msg)
                if logger:
                    logger.info(info_msg)

                raise RuntimeError(f"调用 {gpt_version} 模型失败多次，终止请求。错误信息：{e}")

def cal_cost(response_json, model_name, stage_name=None, total_accumulated_cost=None):
    """
    统计 token 成本
    """
    # 每百万token / 元
    model_cost = {
        # deepseek # 来自o3.fan
        "deepseek-ai/DeepSeek-R1":{"input": 2.80, "output": 11.20},
        "deepseek-ai/DeepSeek-V3":{"input": 1.12, "output": 4.48},
        
        # chatgpt
        "gpt-5-chat-latest":{"input": 8.75, "output": 70.00},
        "chatgpt-4o":{"input": 17.50, "output": 70.00},
        
        # gemini # 来自o3.fan
        "gemini-2.5-pro-preview-05-06": {"input": 8.75, "output": 43.75},
        "gemini-2.0-flash": {"input": 0.70, "output": 2.80},
        "gemini-1.5-flash": {"input": 1.05, "output": 4.20},
        "gemini-1.5-pro": {"input": 2.80, "output": 11.20},
        
        # claude # 来自o3.fan
        "claude-sonnet-4-20250514": {"input": 21.00, "output": 105.00},
        "claude-opus-4-20250514": {"input": 105.00, "output": 525.00},
        "claude-3-7-sonnet-20250219": {"input": 21.00, "output": 105.00},
        
        # Qwen # 来自 Qwen 官方
        "qwen-plus": {"input": 0.80, "output":2.00},  # no-thinking(默认不开启思考模型)
        "qwen-plus-thinking": {"input": 0.80, "output":8.00},  # thinking
    }
    
    if total_accumulated_cost is None:
        total_accumulated_cost = {}

    # 获取输入输出 token 消耗
    input_tokens = response_json["usage"]["prompt_tokens"]
    output_tokens = response_json["usage"]["completion_tokens"]

    # 计算价格
    cost_info = model_cost[model_name]
    input_cost = (input_tokens / 1_000_000) * cost_info['input']
    output_cost = (output_tokens / 1_000_000) * cost_info['output']

    total_cost = input_cost + output_cost
    
    stage_name_set = (
        'process_query',            # 处理用户 query
        'selector',                 # rerank 打分
        'generate_outline',         # 撰写大纲
        'merge_outline',            # 合并大纲
        'generate_report',          # 撰写报告正文
        'check_report_citation',    # 检查报告引用
        'total',                    # 总消耗
        'unnamed_stage',            # 未指定 stage 名称
        )
    
    # 更新阶段数据
    def update_stage_cost(stage_key):
        if stage_key not in total_accumulated_cost and stage_key in stage_name_set:
            total_accumulated_cost[stage_key] = {
                "input_tokens": 0,
                "input_cost": 0.0,
                "cached_tokens": 0,
                "cached_input_cost": 0.0,
                "output_tokens": 0,
                "output_cost": 0.0,
                "total_cost": 0.0 
            }
        
        total_accumulated_cost[stage_key]["input_tokens"] += input_tokens
        total_accumulated_cost[stage_key]["input_cost"] += input_cost
        total_accumulated_cost[stage_key]["output_tokens"] += output_tokens
        total_accumulated_cost[stage_key]["output_cost"] += output_cost
        total_accumulated_cost[stage_key]["total_cost"] += total_cost

        if stage_key != 'total':
            total_accumulated_cost[stage_key]["model_name"] = model_name
    
    # 记录当前 stage 的消耗
    if stage_name:
        update_stage_cost(stage_name)
    # 未知 stage 记录到 unnamed_stage
    else:
        update_stage_cost("unnamed_stage")
    
    # 同步记录到 total stage
    update_stage_cost("total")

    return total_accumulated_cost

def generate_prompt(question, answer):
    prompt = f"""你是一名语言风格与内容分析专家。你的任务是根据提供的问题和回答内容，判断该回答更有可能来自人类还是AI。
请严格遵守以下要求：
- 请仅基于回答的语言风格、逻辑、表达习惯等特征进行判断。
- 根据判断结果，仅回答 "human" 或 "ai"，不得回复其他答案。
- 不得添加任何解释或附加信息。

现在请根据下方的问题与回答对进行判断：
【问题】：
{question}

【回答】：
{answer}"""
    return prompt

def extract_answer(text):
    """
    提取大模型的回答，返回一个元组，第一位：
    0：human
    1：ai
    2：其他回答
    """
    if not text.strip():
        return (2, "None")
    
    if "human" in text.lower() or "man" in text.lower() or "人类" in text:
        return (0, text.strip())
    elif "ai" in text.lower() or "artificial" in text.lower() or  "intelligence" in text.lower():
        return (1, text.strip())
    else:
        return (2, text.strip())
    

def load_model_and_tokenizer(model_path, torch_dtype=torch.float16):
    """
    一次性加载模型和分词器到显存。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch_dtype,
        trust_remote_code=True
    )
    return model, tokenizer

def infer(
        model, 
        tokenizer, 
        system_prompt="你是一个文本来源分析专家助手，需要严格按照我的指令并结合你的专业知识来回答问题。", 
        user_prompt=None,
        max_new_tokens=3
        ):
    """
    使用已加载的模型进行单次推理。
    """
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # 将 message 处理成模型训练时习惯的格式
    input_text = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False
    )
    # encode
    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    ).to(model.device)
    # infer
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # 强制截断
        do_sample=False
    )
    # decode
    output_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return output_text

if __name__ == "__main__":
    MODEL_PATH = ""
    
    # 加载模型到显存
    print("开始加载模型到显存...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    print("模型加载完毕！")
    
    data = [
        {"Q": "0是偶数还是奇数？", "A": "0被2整除后是0，而0是一个整数，因此0是偶数"},
        {"Q": "今天是几月几号？", "A": "对不起，我无法获取实时的日期等信息，您可以在互联网上查询今天的日期信息"}
    ]
    
    for d in data:
        q = d["Q"]
        a = d["A"]
        prompt = generate_prompt(q, a)
        print(prompt + "\n")
        
        response = infer(model, tokenizer, user_prompt=prompt, max_new_tokens=3)
        print(response + "\n")
        
        text = extract_answer(response)
        print(text)
        print("\n")
    