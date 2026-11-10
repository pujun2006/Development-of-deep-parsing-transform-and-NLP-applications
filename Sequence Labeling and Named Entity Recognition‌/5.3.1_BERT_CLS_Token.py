import torch
from transformers import BertModel, BertTokenizer

# 1. 初始化 BERT 的预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 2. 输入文本，进行分词和编码
text = 'John Doe lives in New York and works at OpenAI.'
encoded_input = tokenizer(text, return_tensors='pt')

with torch.no_grad():
  outputs = model(**encoded_input)

# 3. 提取特征向量
cls_output = outputs.last_hidden_state[0, 0]  # [CLS] 向量 (768,)
token_outputs = outputs.last_hidden_state[0]  # 所有 Token 的向量 (seq_len, 768)

print('CLS Output Shape:', cls_output.shape)
print('Token Outputs Shape:', token_outputs.shape)

# 4. 分析每个 Token 的嵌入向量
token_embeddings = []
input_ids_list = encoded_input['input_ids'][0]

print('\n--- 各 Token 详细向量前 5 维展示 ---')
for i, token_id_tensor in enumerate(input_ids_list):
  # 核心修复 1：使用 .item() 转换为 Python int
  token_id = token_id_tensor.item()
  token = tokenizer.convert_ids_to_tokens(token_id)

  # 获取 NumPy 数组
  vec = token_outputs[i].detach().numpy()

  # 核心修复 2：用列表保存元组 (索引, Token字符串, 向量)，防止重复词在字典中被覆盖
  token_embeddings.append((i, token, vec))

  print(f'Index {i}: Token = {token:<12} Embedding[0:5] = {vec[:5]}')

# 5. 验证 [CLS] 与特定单词 'john' 的向量
print('\n--- 向量比对验证 ---')
print('CLS Vector (first 5 elements):', cls_output[:5].numpy())

# 从列表中匹配第一个 'john'
john_item = next(item for item in token_embeddings if item[1] == 'john')
print(
    f"Sample Token Vector - '{john_item[1]}' (Index {john_item[0]}):",
    john_item[2][:5],
)