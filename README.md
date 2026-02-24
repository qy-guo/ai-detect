# LoRA Fine-tuning for AI-Generated Text Detection

基于 HC3 数据集对 Qwen1.5-7B 模型进行 LoRA/QLoRA 微调，用于检测文本是否由 AI 生成。同时使用 GPT-5 生成不同拟人化强度的测试集，评估模型在对抗性样本上的鲁棒性。

## 项目概述

本项目旨在训练一个能够准确区分人类和 AI 生成文本的分类器。主要特点：

- **基础模型**: Qwen1.5-7B-Chat
- **微调方法**: LoRA 和 QLoRA
- **训练数据**: HC3-Chinese 数据集（包含百科、经济、法律、医学四个领域）
- **测试数据**:
  - 原始 HC3 测试集（ChatGPT 生成的回答）
  - GPT-5 生成的三种拟人化强度的测试集
    - **Origin**: 无拟人化指令的原始 GPT-5 回答
    - **Mid**: 中等强度的拟人化回答
    - **Strong**: 强拟人化回答

## 目录结构

```
LoRA/
├── dataset/                          # 数据集相关
│   ├── AI-ModelScope/HC3-Chinese/    # HC3原始数据集
│   │   ├── baike.jsonl              # 百科领域 (4617条)
│   │   ├── finance.jsonl            # 经济领域 (689条)
│   │   ├── law.jsonl                # 法律领域 (372条)
│   │   └── medicine.jsonl           # 医学领域 (1074条)
│   ├── dataset.py                   # 数据集划分与处理
│   ├── clean_dataset.py             # 数据清洗脚本
│   └── process_gpt5_testdata.py     # GPT-5测试集生成
├── output/                          # 模型输出目录
│   ├── lora_*/                      # LoRA微调模型检查点
│   ├── qlora_*/                     # QLoRA微调模型检查点
│   └── baseline/                    # 基线模型评估结果
├── result/                          # 实验结果文档
├── train_lora.py                    # LoRA训练脚本
├── train_qlora.py                   # QLoRA训练脚本
├── test_lora_vllm.py                # LoRA模型测试（使用vLLM加速）
├── test_qlora_vllm.py               # QLoRA模型测试（使用vLLM加速）
├── baseline.py                      # 基线模型评估
├── baseline_vllm.py                 # 基线模型评估（vLLM加速）
└── utils_wo_apikey.py               # 工具函数（已移除API密钥）
```

## 环境配置

### 依赖安装

```bash
pip install torch transformers datasets peft trl
pip install vllm  # 用于加速推理
pip install bitsandbytes  # QLoRA量化所需
pip install scikit-learn  # 评估指标
```

## 使用方法

### 1. 数据准备

#### 1.1 处理 HC3 数据集

```bash
cd dataset
python dataset.py
```

该脚本会：
- 加载 HC3-Chinese 的四个领域数据
- 按 8:1:1 比例划分训练/验证/测试集
- 将每个问题拆分为 `human` 和 `ai` 两个问答对
- 输出到 `dataset/processed/` 目录

#### 1.2 生成 GPT-5 测试集（可选）

```bash
cd dataset
python process_gpt5_testdata.py
```

该脚本会：
- 读取原始测试集
- 使用 GPT-5 生成三种不同拟人化强度的回答
- 保持与人类回答相同的字数和句子数
- 支持并发生成以提高效率

#### 1.3 数据清洗

```bash
cd dataset
python clean_dataset.py
```

清洗规则：
- 移除空白字符和换行符
- 删除回答为空的问答对
- 删除关联问题的所有问答对（保持数据一致性）

### 2. 模型训练

#### 2.1 LoRA 微调

```bash
python train_lora.py
```

**配置参数**:
- `r=8`: LoRA秩
- `lora_alpha=16`: 缩放因子
- `lora_dropout=0.1`: Dropout率
- `num_train_epochs=5`: 训练轮数
- `per_device_train_batch_size=2`: 批次大小
- `gradient_accumulation_steps=4`: 梯度累积步数
- `learning_rate=1e-5`: 学习率

**目标模块**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

#### 2.2 QLoRA 微调

```bash
python train_qlora.py
```

**量化配置**:
- `load_in_4bit=True`: 4-bit量化
- `bnb_4bit_quant_type="nf4"`: NF4量化类型
- `bnb_4bit_compute_dtype=torch.bfloat16`: 计算时使用bf16
- `bnb_4bit_use_double_quant=True`: 双重量化

**其他参数**: 与 LoRA 相同

### 3. 模型评估

#### 3.1 使用 vLLM 加速推理

```bash
# 评估 LoRA 模型
python test_lora_vllm.py

# 评估 QLoRA 模型
python test_qlora_vllm.py
```

#### 3.2 基线模型评估

```bash
# 标准推理
python baseline.py

# vLLM加速推理
python baseline_vllm.py
```

### 4. 评估指标

模型在四个测试集上进行评估，输出以下指标：

- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率（TPR）
- **Specificity**: 特异度（TNR）
- **MCC**: Matthews相关系数
- **Confusion Matrix**: 混淆矩阵

评估结果会被分类保存为：
- `TP_AI_is_AI.jsonl`: 正确识别为 AI
- `TN_Human_is_Human.jsonl`: 正确识别为 Human
- `FP_Human_is_AI.jsonl`: Human 误判为 AI
- `FN_AI_is_Human.jsonl`: AI 误判为 Human

## Prompt 设计

### 训练 Prompt

```python
system_prompt = "你是一个文本来源分析专家助手，需要严格按照我的指令并结合你的专业知识来回答问题。"

user_prompt = """你是一名语言风格与内容分析专家。你的任务是根据提供的问题和回答内容，判断该回答更有可能来自人类还是AI。
请严格遵守以下要求：
- 请仅基于回答的语言风格、逻辑、表达习惯等特征进行判断。
- 根据判断结果，仅回答 "human" 或 "ai"，不得回复其他答案。
- 不得添加任何解释或附加信息。

现在请根据下方的问题与回答对进行判断：
【问题】：
{question}

【回答】：
{answer}"""
```

### GPT-5 生成 Prompt

根据拟人化强度分为三个等级：

1. **Origin**: 技术写作风格，无拟人化指令
2. **Mid**: 避免 AI 口癖，自然叙述
3. **Strong**: 强烈要求模仿人类口吻

所有生成都要求：
- 控制字数和句子数与人类回答一致
- 不引入外部资料或新事实
- 避免"感谢提问/作为AI/抱歉/我不能"等 AI 标志性短语


## 数据集信息

### HC3-Chinese 数据集

来源：[AI-ModelScope/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)

引用：
```bibtex
@article{guo-etal-2023-hc3,
    title = "How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection",
    author = "Guo, Biyang and Zhang, Xin and Wang, Ziyuan and Jiang, Minqi and
              Nie, Jinran and Ding, Yuxuan and Yue, Jianwei and Wu, Yupeng",
    journal = {arXiv preprint arxiv:2301.07597},
    year = "2023",
}
```

### 数据格式

```json
{
  "question": "我有一个计算机相关的问题，请用中文回答，什么是硬盘安装",
  "human_answers": ["硬盘安装就是从硬盘安装XP的系统..."],
  "chatgpt_answers": ["硬盘安装是指将软件或操作系统安装到硬盘上的过程..."]
}
```

处理后格式：
```json
{
  "question": "...",
  "answer": "...",
  "source": "human"  // 或 "ai"
}
```

## 实验结果

详细的实验结果和分析请参见 `result/` 目录中的文档。

## 注意事项

1. **API 密钥**: `utils.py` 包含 API 密钥，已添加到 `.gitignore`。请使用 `utils_wo_apikey.py` 作为参考。
2. **路径配置**: 训练和测试脚本中的路径需要根据实际环境修改。
3. **显存管理**: 如果遇到 OOM 错误，可以：
   - 减小 `per_device_train_batch_size`
   - 增加 `gradient_accumulation_steps`
   - 使用 QLoRA 替代 LoRA
   - 减小 `max_model_len`
4. **数据打乱**: 训练前已对数据集进行打乱（`seed=42`），确保随机性。

## 许可证

- 代码：MIT License
- HC3 数据集：CC-BY-SA 4.0

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) 提供的基础模型
- [HC3](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection) 数据集
- [PEFT](https://github.com/huggingface/peft) 和 [TRL](https://github.com/huggingface/trl) 库
- [vLLM](https://github.com/vllm-project/vllm) 高性能推理引擎