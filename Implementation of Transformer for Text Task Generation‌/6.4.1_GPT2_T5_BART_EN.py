import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
)

# 1. 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 定义模型和分词器（修复 1：将 facebook/bart-small 替换为官方正确的 facebook/bart-base）
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# 修复 2：为 GPT-2 补全 pad_token，消除控制台警告
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

bart_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-base"
).to(device)
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# 输入文本示例
input_text = "In a world where technology advances rapidly"

# 3. GPT-2 生成 (自回归文本续写)
gpt2_input = gpt2_tokenizer(input_text, return_tensors="pt").to(device)
gpt2_output = gpt2_model.generate(
    **gpt2_input,
    max_length=50,
    num_return_sequences=1,
    pad_token_id=gpt2_tokenizer.eos_token_id,
)
gpt2_generated_text = gpt2_tokenizer.decode(
    gpt2_output[0], skip_special_tokens=True
)

# 4. T5 生成 (基于 Task-Prefix 的条件文本生成)
t5_input = t5_tokenizer("summarize: " + input_text, return_tensors="pt").to(
    device
)
# 修复 3：传入完整张量字典（包含 input_ids 和 attention_mask）
t5_output = t5_model.generate(
    **t5_input, max_length=50, num_return_sequences=1
)
t5_generated_text = t5_tokenizer.decode(
    t5_output[0], skip_special_tokens=True
)

# 5. BART 生成 (Seq2Seq 重构生成/摘要)
bart_input = bart_tokenizer(input_text, return_tensors="pt").to(device)
# 修复 3：传入完整张量字典（包含 input_ids 和 attention_mask）
bart_output = bart_model.generate(
    **bart_input, max_length=50, num_return_sequences=1
)
bart_generated_text = bart_tokenizer.decode(
    bart_output[0], skip_special_tokens=True
)

# 6. 打印对比生成结果
print("=== GPT-2 Generated Text ===")
print(gpt2_generated_text)

print("\n=== T5 Generated Summary ===")
print(t5_generated_text)

print("\n=== BART Generated Text ===")
print(bart_generated_text)