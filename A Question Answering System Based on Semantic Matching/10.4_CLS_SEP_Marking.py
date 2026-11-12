import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# 1. 加载已经在 SQuAD 上微调过的问答模型
model_name = "csarron/bert-base-uncased-squad-v1"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# 2. 定义问题和上下文
question = "What is the capital of France?"
context = "France is a country in Europe. Its capital is Paris, which is known for its culture and history."

# 3. 标准输入方式：直接将 question 和 context 作为前两个参数传入
# tokenizer 会自动为你添加 [CLS]、[SEP] 以及生成 token_type_ids
inputs = tokenizer(question, context, return_tensors="pt")

print("编码后的输入 Token:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# 4. 获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# 5. 找到答案的起始和结束位置
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits) + 1

# 6. 解析出答案（推荐使用 decode 方法，更干净）
answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index], skip_special_tokens=True)

print("\n问题:", question)
print("预测答案:", answer)