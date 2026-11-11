import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

model_name = "xlm-roberta-base"

# 1. 使用通用 Auto 类初始化 Tokenizer 和分类模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

texts = ["Hello, how are you?", "Bonjour, comment ça va?", "Hola, ¿cómo estás?"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# =========================================================
# 任务一：多语言文本分类预测
# =========================================================
print("--- 1. 多语言文本分类预测 ---")
with torch.no_grad():
  outputs = classification_model(**inputs, output_hidden_states=True)
  predictions = F.softmax(outputs.logits, dim=-1)

for idx, text in enumerate(texts):
  print(f"Text: {text}")
  print("Predicted Probabilities:", predictions[idx].numpy().round(4))
print("\n")

# =========================================================
# 任务二：提取跨语言通用句向量（使用正确的 Masked Mean Pooling）
# =========================================================
print("--- 2. 提取跨语言文本语义向量 (Sentence Embeddings) ---")
# 直接复用 classification_model 内部的 roberta backbone，无需重复加载模型
last_hidden_state = outputs.hidden_states[-1]

# 结合 attention_mask 进行正确的平均池化（排除 padding 干扰）
attention_mask = inputs["attention_mask"].unsqueeze(-1)
sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
sentence_embeddings = sum_embeddings / sum_mask

for idx, text in enumerate(texts):
  print(f"Text: {text}")
  print(
      "Sentence Vector (first 5 dims):",
      sentence_embeddings[idx, :5].numpy().round(4),
  )
print("\n")

# =========================================================
# 任务三：真正的多语言机器翻译（使用专用的 Seq2Seq 翻译模型）
# =========================================================
print("--- 3. 真正的机器翻译示例 (法文 -> 中文) ---")
trans_model_name = "facebook/m2m100_418M"
trans_tokenizer = M2M100Tokenizer.from_pretrained(trans_model_name)
trans_model = M2M100ForConditionalGeneration.from_pretrained(trans_model_name)

src_text = "Bonjour, comment ça va?"
trans_tokenizer.src_lang = "fr"  # 设置源语言为法文

encoded_input = trans_tokenizer(src_text, return_tensors="pt")
generated_tokens = trans_model.generate(
    **encoded_input,
    forced_bos_token_id=trans_tokenizer.get_lang_id("zh"),  # 设置目标语言为中文
)
translated_text = trans_tokenizer.batch_decode(
    generated_tokens, skip_special_tokens=True
)[0]

print(f"源文本 (法文): {src_text}")
print(f"翻译结果 (中文): {translated_text}")