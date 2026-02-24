from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_path = r"/storage/public/liutg/qyguo/qyguo_llm/LLM/models/qwen1.5-7b-Chat"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    dtype=torch.bfloat16)

block0 = model.model.layers[0]

for name, modules in block0.named_modules():
    print(name, type(modules))