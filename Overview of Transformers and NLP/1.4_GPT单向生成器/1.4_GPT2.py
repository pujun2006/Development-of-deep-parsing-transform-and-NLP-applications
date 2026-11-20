import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载GPT模型和分词器
gpt_model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
model = GPT2LMHeadModel.from_pretrained(gpt_model_name)

# 输入文本并进行分词
input_text = "Once upon a time in a distant land,"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 使用GPT生成文本
output_sequences = model.generate(
    input_ids=input_ids,
    max_length=50,         # 设置生成文本的最大长度
    num_return_sequences=1, # 返回的生成序列数
    no_repeat_ngram_size=2, # 防止生成重复的短语
    top_k=50,               # 限制每次生成的单词数量
    top_p=0.95,             # 采用nucleus sampling
    temperature=0.7,        # 控制生成文本的随机性
)

# 解码生成的输出文本
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# 输出结果
print("GPT模型生成的文本:")
print(generated_text)
