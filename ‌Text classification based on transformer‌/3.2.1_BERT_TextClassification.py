from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer

# 1. 数据准备
texts = [
    "人工智能在金融市场中的应用日益广泛",
    "足球和篮球是全球最受欢迎的运动",
    "股市波动对投资者有很大影响",
    "机器学习和数据科学的结合推动了科技创新",
    "外汇市场的变化使得投资回报更具挑战性",
    "网球比赛十分激烈，场上选手表现出色",
    "数据分析在金融行业中非常重要",
    "篮球赛事中三分球表现出色",
]
labels = [1, 0, 1, 1, 1, 0, 1, 0]  # 1表示财经/科技，0表示体育

# 分割训练和测试集（使用 stratify 保证测试集包含两个类别）
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)


# 2. 自定义数据集
class TextDataset(Dataset):

  def __init__(self, texts, labels, tokenizer, max_length=64):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    encoding = self.tokenizer(
        self.texts[idx],
        padding="max_length",
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt",
    )
    label = torch.tensor(self.labels[idx], dtype=torch.long)
    return (
        encoding["input_ids"].squeeze(0),
        encoding["attention_mask"].squeeze(0),
        label,
    )


# 3. 加载中文 BERT 模型和分词器（关键修改：改为 bert-base-chinese）
MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# 4. 数据加载器
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# 5. 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)


# 6. 训练函数（增加多 Epoch 训练）
def train_model(model, data_loader, optimizer, device, epochs=5):
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for input_ids, attention_mask, labels in data_loader:
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(
          input_ids=input_ids, attention_mask=attention_mask, labels=labels
      )
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs} - 平均训练损失:"
        f" {total_loss / len(data_loader):.4f}"
    )


# 7. 评估函数
def evaluate_model(model, data_loader, device):
  model.eval()
  predictions, true_labels = [], []
  with torch.no_grad():
    for input_ids, attention_mask, labels in data_loader:
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      logits = outputs.logits
      preds = torch.argmax(logits, dim=1)

      predictions.extend(preds.cpu().numpy())
      true_labels.extend(labels.numpy())

  acc = accuracy_score(true_labels, predictions)
  print("\n=== 测试集评估结果 ===")
  print(f"测试准确率: {acc:.2f}")
  print(
      classification_report(
          true_labels,
          predictions,
          target_names=["体育(0)", "财经/科技(1)"],
          zero_division=0,
      )
  )


# 8. 执行训练与测试
train_model(model, train_loader, optimizer, device, epochs=5)
evaluate_model(model, test_loader, device)
