import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5' # GPU
import torch
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    HfArgumentParser,
    BitsAndBytesConfig   # !! QLoRA核心：导入BitsAndBytesConfig !!
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from functools import partial


# 加载配置模型、分词器
print("开始加载模型和分词器...")
model_path = r"/storage/public/liutg/qyguo/qyguo_llm/LLM/models/qwen1.5-7b-Chat"

# !! QLoRA核心：定义4-bit量化配置 !!
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                # 使用NF4 (Normal Float 4) 进行量化，这是一种专为正态分布权重优化的格式
    bnb_4bit_compute_dtype=torch.bfloat16,    # 在计算过程中，权重会被反量化成bf16格式进行矩阵乘法，以保证精度
    bnb_4bit_use_double_quant=True,           # 启用双重量化，对量化常数本身再次进行量化，进一步节省显存
)

# !! QLoRA核心：使用量化配置加载模型 !!
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config, # 传入量化配置
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("模型和分词器加载完毕。")



# 加载数据集
def format_dataset_with_template(example, tokenizer):
    """
    处理数据集，保证与 baseline 一致
    """
    system_prompt="你是一个文本来源分析专家助手，需要严格按照我的指令并结合你的专业知识来回答问题。"
    prompt = f"""你是一名语言风格与内容分析专家。你的任务是根据提供的问题和回答内容，判断该回答更有可能来自人类还是AI。
    请严格遵守以下要求：
    - 请仅基于回答的语言风格、逻辑、表达习惯等特征进行判断。
    - 根据判断结果，仅回答 "human" 或 "ai"，不得回复其他答案。
    - 不得添加任何解释或附加信息。

    现在请根据下方的问题与回答对进行判断：
    【问题】：
    {example["question"]}

    【回答】：
    {example["answer"]}"""
    
    # 构建 messages 列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example['source']} # "human" or "ai"
    ]
    
    # !! 最终修复：直接进行分词，生成模型输入 !!
    model_inputs = tokenizer(
        tokenizer.apply_chat_template(messages, tokenize=False),
        max_length=512,
        truncation=True,
        padding=False, # DataCollator会处理padding
    )
    return model_inputs

print("开始加载和处理数据集...")
data_dir = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/dataset/processed/clean"
train_data_raw = load_dataset(path="json", data_files=os.path.join(data_dir, "train.jsonl"), split="train")
valid_data_raw = load_dataset(path="json", data_files=os.path.join(data_dir, "valid.jsonl"), split="train")

# !! 核心修改：在处理数据前打乱训练集 !!
print("正在打乱训练集...")
train_data_raw = train_data_raw.shuffle(seed=42)

# 使用全量数据
print(f"已选取 {len(train_data_raw)} 个样本进行训练。")

# 使用partial来固定tokenizer参数
formatting_function = partial(format_dataset_with_template, tokenizer=tokenizer)

print("正在预处理和分词训练集...")
train_dataset = train_data_raw.map(formatting_function, remove_columns=list(train_data_raw.features))
print("正在预处理和分词验证集...")
valid_dataset = valid_data_raw.map(formatting_function, remove_columns=list(valid_data_raw.features))

# 打印一个样本检查格式是否正确
print("检查处理后的第一个样本的keys：")
print(train_dataset[0].keys())
print("数据集加载和处理完毕。")


# 配置 LoRA (与LoRA脚本完全相同)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",      # 任务类型，文本生成
    target_modules=[            # 指定 LoRA 插入的位置
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    r=8,                        # 秩大小
    lora_alpha=16,              # 缩放因子，通常为 r 的两倍，(alpha / r) * B * A * x
    lora_dropout=0.1,           # dropout(B * A * x)
    bias="none"
)
print("LoRA 配置完成。")



# 配置训练参数
output_path = r"/storage/public/liutg/qyguo/qyguo_llm/LoRA/output/qlora_2"
training_args = TrainingArguments(
    output_dir=output_path,         # 输出文件路径
    num_train_epochs=5,             # epoch
    per_device_train_batch_size=2,  # mini-batch
    gradient_accumulation_steps=4,  # step
    learning_rate=1e-5,             # lr
    max_grad_norm=0.3,              # 梯度裁剪，单步梯度变化上限
    
    eval_strategy="steps",          # 按步数进行评估
    save_strategy="steps",          # 按步数进行保存 (与评估策略保持一致)
    eval_steps=100,                 # 每n步在验证集上进行验证
    save_steps=100,                 # 每n步保存一次模型检查点
    logging_steps=20,               # 每20次记录 logging
    
    bf16=True,                      # 使用 bf16 进行训练
    tf32=True,                      # 开启 tf32 格式进行计算
    report_to="tensorboard",        # 将日志报告给 tensorboard
    
    load_best_model_at_end=True,    # 训练结束后加载性能最好的模型
    metric_for_best_model="loss",   # 使用验证集损失(eval_loss)作为衡量最佳模型的指标
    # save_total_limit=10,            # 最多保留新检查点数
)
print("训练参数配置完成。")


from transformers import DataCollatorForLanguageModeling

# 定义 SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,    
    peft_config=peft_config,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),    # 如果连续3次评估，验证损失都没下降，就停止
    ]
)
print("Trainer 初始化完成。")



# 训练并保存模型权重
print("开始模型训练...")
trainer.train()
print("模型训练完成。")

final_model_path = os.path.join(output_path, "final_model")
trainer.save_model(final_model_path)
print(f"模型已保存至: {final_model_path}")
