import os
import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
)

# 开启国内镜像源加速模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载【中文预训练模型】与分词器（替换为中文模型）
# T5 中文模型 (孟子中文 T5)
t5_model_name = "Langboat/mengzi-t5-base"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

# GPT-2 中文模型
gpt2_model_name = "uer/gpt2-chinese-cluecorpussmall"
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # 补充 pad_token

# BART 中文模型
bart_model_name = "fnlp/bart-base-chinese"
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(
    device
)

# 2. 中文输入文本
input_text = "科技快速发展的今天，人们的生活方式发生了巨大的改变，关注健康和环境成为新趋势。"

# 3. T5 模型生成
t5_input = t5_tokenizer(input_text, return_tensors="pt").to(device)
t5_input.pop("token_type_ids", None)  # 移除 T5 不支持的 token_type_ids

t5_output = t5_model.generate(
    **t5_input, max_length=50, num_beams=4, early_stopping=True
)
# decode 时加上 .replace(" ", "") 移除中文 Tokenizer 默认产生的字间空格
t5_summary = t5_tokenizer.decode(
    t5_output[0], skip_special_tokens=True
).replace(" ", "")

# 4. GPT-2 模型生成
gpt2_input = gpt2_tokenizer(input_text, return_tensors="pt").to(device)
gpt2_output = gpt2_model.generate(
    **gpt2_input,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,  # 惩罚重复词
    pad_token_id=gpt2_tokenizer.eos_token_id,
)
gpt2_dialogue = gpt2_tokenizer.decode(
    gpt2_output[0], skip_special_tokens=True
).replace(" ", "")

# 5. BART 模型生成
bart_input = bart_tokenizer(input_text, return_tensors="pt").to(device)
bart_input.pop("token_type_ids", None)

bart_output = bart_model.generate(
    **bart_input, max_length=50, num_beams=4, early_stopping=True
)
bart_summary = bart_tokenizer.decode(
    bart_output[0], skip_special_tokens=True
).replace(" ", "")

# 6. 打印生成结果
print("=== T5模型生成的文本摘要 ===")
print(t5_summary)
print("\n=== GPT-2模型生成的续写内容 ===")
print(gpt2_dialogue)
print("\n=== BART模型生成的文本摘要 ===")
print(bart_summary)