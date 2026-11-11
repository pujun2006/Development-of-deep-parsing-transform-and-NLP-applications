import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

# 1. 定义领域词典（转换为小写以便与 uncased tokenizer 匹配）
gazetteer = {
    "location": ["paris", "new york", "beijing", "london"],
    "organization": ["google", "microsoft", "openai", "tesla"],
    "person": ["john", "alice", "bob", "charlie"],
}

# 全局初始化 Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 2. 数据集类
class TextDataset(Dataset):

  def __init__(self, texts, tokenizer):
    self.texts = texts
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = self.texts[idx]
    encoding = self.tokenizer(
        text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # 将字典扁平化处理，去掉 batch 维度 ([1, 64] -> [64])
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "text": text,
    }


# 3. 模型定义
class NERModel(nn.Module):

  def __init__(self):
    super(NERModel, self).__init__()
    self.bert = BertModel.from_pretrained("bert-base-uncased")
    self.classifier = nn.Linear(
        self.bert.config.hidden_size, 4
    )  # 0: O, 1: location, 2: organization, 3: person

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = outputs.last_hidden_state
    logits = self.classifier(sequence_output)
    return logits


# 4. 词典特征匹配函数（支持 Token 与简单子词匹配）
def integrate_gazetteer(tokens, gazetteer):
  gazetteer_mask = torch.zeros(len(tokens), dtype=torch.long)
  # 展平词典列表进行搜索
  for idx, token in enumerate(tokens):
    clean_token = token.replace("##", "").lower()
    if clean_token in ["[cls]", "[sep]", "[pad]"]:
      continue

    for ent_type_idx, (entity_type, words) in enumerate(
        gazetteer.items(), start=1
    ):
      for word in words:
        if clean_token in word.split():
          gazetteer_mask[idx] = ent_type_idx
          break
  return gazetteer_mask


# 5. 初始化与数据准备
texts = ["John works at Google in New York.", "Alice visited Paris last year."]
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = NERModel()
model.eval()

# 6. 执行预测与匹配展示
print("=== 开始处理与匹配 ===")
for batch in dataloader:
  input_ids = batch["input_ids"]
  attention_mask = batch["attention_mask"]
  text = batch["text"][0]

  # 关键修复：使用 tokenizer.convert_ids_to_tokens 替代 model.bert.config.id2token
  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

  # 将词典信息整合到输入中
  gazetteer_mask = integrate_gazetteer(tokens, gazetteer)

  with torch.no_grad():
    logits = model(input_ids, attention_mask)

  predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()

  # 整理并输出结果
  result = []
  for token, pred, gaz_mask in zip(tokens, predictions, gazetteer_mask):
    if token in ["[PAD]", "[CLS]", "[SEP]"]:
      continue
    # 只要模型预测非 0 或词典匹配非 0，即输出显示
    if pred != 0 or gaz_mask.item() != 0:
      result.append((
          token,
          f"Pred:{pred}",
          f"GazetteerMatch:{gaz_mask.item()}",
      ))

  print(f"\nText: {text}")
  print("Tokens & Gazetteers Match Result:", result)