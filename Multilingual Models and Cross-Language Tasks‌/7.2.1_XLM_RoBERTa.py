import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

# 定义模型名称
model_name = "xlm-roberta-base"

# 1. 统一初始化 Tokenizer 和 Model (避免重复加载)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# 定义多语言输入文本
texts = ["Hello, how are you?", "Bonjour, comment ça va?", "Hola, ¿cómo estás?"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# =========================================================
# 任务一：提取多语言文本的句向量表示 (Mean Pooling)
# =========================================================
print("--- 1. 提取多语言语义向量 ---")
with torch.no_grad():
  outputs = model(**inputs)
  # last_hidden_state: [batch_size, seq_len, hidden_dim]
  token_embeddings = outputs.last_hidden_state

  # 计算句子粒度的 Mean Pooling 向量 (忽略 Padding 影响)
  attention_mask = inputs["attention_mask"].unsqueeze(-1)
  sentence_embeddings = torch.sum(
      token_embeddings * attention_mask, dim=1
  ) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)

print("Sentence Embeddings Shape:", sentence_embeddings.shape)
for idx, text in enumerate(texts):
  print(f"Text: {text}")
  print(
      "Embedding (first 5 dims):",
      sentence_embeddings[idx, :5].numpy().round(4),
  )
print("\n")

# =========================================================
# 任务二：多语言序列分类任务 (Sequence Classification)
# =========================================================
print("--- 2. 多语言文本分类预测 ---")
with torch.no_grad():
  class_outputs = classification_model(**inputs)
  predictions = F.softmax(class_outputs.logits, dim=-1)

for idx, text in enumerate(texts):
  print(f"Text: {text}")
  print(
      "Predicted Class Probabilities:", predictions[idx].numpy().round(4)
  )
print("\n")

# =========================================================
# 任务三：真正的多语言机器翻译 (使用 M2M100 翻译模型)
# =========================================================
print("--- 3. 真正的多语言机器翻译 (法文/西班牙文 -> 中文) ---")
trans_model_name = "facebook/m2m100_418M"
trans_tokenizer = M2M100Tokenizer.from_pretrained(trans_model_name)
trans_model = M2M100ForConditionalGeneration.from_pretrained(trans_model_name)

# 以第二句法文 "Bonjour, comment ça va?" 翻译为中文为例
src_text = "Bonjour, comment ça va?"
trans_tokenizer.src_lang = "fr"  # 设置源语言为法文

encoded_zh = trans_tokenizer(src_text, return_tensors="pt")
generated_tokens = trans_model.generate(
    **encoded_zh,
    forced_bos_token_id=trans_tokenizer.get_lang_id("zh"),  # 目标语言设为中文
)
translated_text = trans_tokenizer.batch_decode(
    generated_tokens, skip_special_tokens=True
)[0]

print(f"源文本 (法文): {src_text}")
print(f"翻译结果 (中文): {translated_text}")