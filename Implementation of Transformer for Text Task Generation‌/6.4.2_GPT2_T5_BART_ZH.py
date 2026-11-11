# import torch
# from transformers import (
#     AutoTokenizer,
#     BartForConditionalGeneration,
#     GPT2LMHeadModel,
#     T5ForConditionalGeneration,
# )
#
# # 1. 设置设备
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # 2. 定义模型和分词器（修复 1：将 facebook/bart-small 替换为官方正确的 facebook/bart-base）
# gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
# gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# # 修复 2：为 GPT-2 补全 pad_token，消除控制台警告
# gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
#
# t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
# t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
#
# bart_model = BartForConditionalGeneration.from_pretrained(
#     "facebook/bart-base"
# ).to(device)
# bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
#
# # 输入文本示例
# input_text = "In a world where technology advances rapidly"
#
# # 3. GPT-2 生成 (自回归文本续写)
# gpt2_input = gpt2_tokenizer(input_text, return_tensors="pt").to(device)
# gpt2_output = gpt2_model.generate(
#     **gpt2_input,
#     max_length=50,
#     num_return_sequences=1,
#     pad_token_id=gpt2_tokenizer.eos_token_id,
# )
# gpt2_generated_text = gpt2_tokenizer.decode(
#     gpt2_output[0], skip_special_tokens=True
# )
#
# # 4. T5 生成 (基于 Task-Prefix 的条件文本生成)
# t5_input = t5_tokenizer("summarize: " + input_text, return_tensors="pt").to(
#     device
# )
# # 修复 3：传入完整张量字典（包含 input_ids 和 attention_mask）
# t5_output = t5_model.generate(
#     **t5_input, max_length=50, num_return_sequences=1
# )
# t5_generated_text = t5_tokenizer.decode(
#     t5_output[0], skip_special_tokens=True
# )
#
# # 5. BART 生成 (Seq2Seq 重构生成/摘要)
# bart_input = bart_tokenizer(input_text, return_tensors="pt").to(device)
# # 修复 3：传入完整张量字典（包含 input_ids 和 attention_mask）
# bart_output = bart_model.generate(
#     **bart_input, max_length=50, num_return_sequences=1
# )
# bart_generated_text = bart_tokenizer.decode(
#     bart_output[0], skip_special_tokens=True
# )
#
# # 6. 打印对比生成结果
# print("=== GPT-2 Generated Text ===")
# print(gpt2_generated_text)
#
# print("\n=== T5 Generated Summary ===")
# print(t5_generated_text)
#
# print("\n=== BART Generated Text ===")
# print(bart_generated_text)

import os
import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
)

# 1. 开启国内镜像源，加速模型和分词器的下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 设置运行设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device}")

# ==========================================
# 3. 加载模型与 Tokenizer
# ==========================================
print("正在加载模型和分词器...")

# --- GPT-2 模型 ---
gpt2_model_name = "uer/gpt2-chinese-cluecorpussmall"
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # 补充 pad_token，消除警告

# --- T5 模型 (使用稳定开源的 Langboat 孟子中文 T5) ---
t5_model_name = "Langboat/mengzi-t5-base"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

# --- BART 模型 ---
bart_model_name = "fnlp/bart-base-chinese"
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(
    device
)

print("所有模型加载完毕！\n")

# ==========================================
# 4. 定义输入文本与执行生成
# ==========================================
input_text = "在一个科技快速发展的世界中，人们面临着不断变化的生活方式"

# ------------------------------------------
# (1) GPT-2 文本续写
# ------------------------------------------
gpt2_input = gpt2_tokenizer(input_text, return_tensors="pt").to(device)
gpt2_output = gpt2_model.generate(
    **gpt2_input,
    max_length=60,
    do_sample=True,  # 开启采样，增强中文生成的创造力
    top_p=0.9,
    temperature=0.8,
    repetition_penalty=1.2,  # 惩罚重复词，避免陷入死循环
    pad_token_id=gpt2_tokenizer.eos_token_id,
)
# 解码并清除中文字符间的空格
gpt2_generated_text = gpt2_tokenizer.decode(
    gpt2_output[0], skip_special_tokens=True
).replace(" ", "")

# ------------------------------------------
# (2) T5 文本生成 / 摘要 / 重构
# ------------------------------------------
t5_input = t5_tokenizer(input_text, return_tensors="pt").to(device)
# 【关键修复】移除 T5 不需要的 token_type_ids 参数，防止报错 ValueError
t5_input.pop("token_type_ids", None)

t5_output = t5_model.generate(
    **t5_input, max_length=50, num_beams=4, early_stopping=True
)
t5_generated_text = t5_tokenizer.decode(
    t5_output[0], skip_special_tokens=True
).replace(" ", "")

# ------------------------------------------
# (3) BART 文本生成 / 重构
# ------------------------------------------
bart_input = bart_tokenizer(input_text, return_tensors="pt").to(device)
bart_input.pop("token_type_ids", None)  # 同样移除潜在的多余字段

bart_output = bart_model.generate(
    **bart_input, max_length=50, num_beams=4, early_stopping=True
)
bart_generated_text = bart_tokenizer.decode(
    bart_output[0], skip_special_tokens=True
).replace(" ", "")

# ==========================================
# 5. 打印对比生成结果
# ==========================================
print("=" * 40)
print("【原始输入】")
print(input_text)
print("=" * 40)

print("\n=== GPT-2 生成结果 (续写风格) ===")
print(gpt2_generated_text)

print("\n=== T5 (孟子) 生成结果 (提炼/重构风格) ===")
print(t5_generated_text)

print("\n=== BART 生成结果 (重构风格) ===")
print(bart_generated_text)