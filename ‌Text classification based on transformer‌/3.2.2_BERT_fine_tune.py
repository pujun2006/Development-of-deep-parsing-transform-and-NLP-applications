from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer

# 1. 示例数据
texts = [
    "金融市场有很大波动",
    "科技公司股价上涨",
    "体育赛事热度高",
    "旅游行业恢复",
    "股市持续下跌",
    "新的电影上映吸引大量观众",
    "天气变化较大",
    "电影票房破纪录",
    "新科技成果引发关注",
    "环保问题引起重视",
]
binary_labels = [1, 1, 0, 0, 1, 0, 0, 0, 1, 0]  # 1:财经/科技，0:非财经
multi_labels = [0, 0, 1, 2, 0, 1, 2, 1, 0, 2]  # 0:财经/科技，1:娱乐/体育，2:其他

# 分割数据集（使用 stratify 保持标签比例一致）
(
    train_texts,
    test_texts,
    train_bin_labels,
    test_bin_labels,
    train_multi_labels,
    test_multi_labels,
) = train_test_split(
    texts,
    binary_labels,
    multi_labels,
    test_size=0.3,
    random_state=42,
    stratify=multi_labels,
)


# 2. 自定义数据集
class TextDataset(Dataset):

  def __init__(self, texts, labels, tokenizer, max_length=32):
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
    # 使用 squeeze(0) 明确消除 batch 维度
    return (
        encoding["input_ids"].squeeze(0),
        encoding["attention_mask"].squeeze(0),
        label,
    )


# 3. 加载中文 BERT 分词器（关键修改：bert-base-chinese）
MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 创建 Dataset 与 DataLoader
bin_train_loader = DataLoader(
    TextDataset(train_texts, train_bin_labels, tokenizer),
    batch_size=2,
    shuffle=True,
)
bin_test_loader = DataLoader(
    TextDataset(test_texts, test_bin_labels, tokenizer), batch_size=2
)

multi_train_loader = DataLoader(
    TextDataset(train_texts, train_multi_labels, tokenizer),
    batch_size=2,
    shuffle=True,
)
multi_test_loader = DataLoader(
    TextDataset(test_texts, test_multi_labels, tokenizer), batch_size=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 4. 改进后的训练函数（增加多 Epoch 循环）
def train_model(model, data_loader, optimizer, device, epochs=5):
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for input_ids, attention_mask, labels in data_loader:
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      # BertForSequenceClassification 自动计算内部 Loss
      outputs = model(
          input_ids=input_ids, attention_mask=attention_mask, labels=labels
      )
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(
        f"Epoch {epoch+1}/{epochs} - 平均损失:"
        f" {total_loss/len(data_loader):.4f}"
    )


# 5. 改进后的评估函数
def evaluate_model(model, data_loader, device):
  model.eval()
  predictions, true_labels = [], []
  with torch.no_grad():
    for input_ids, attention_mask, labels in data_loader:
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      preds = torch.argmax(outputs.logits, dim=1)

      predictions.extend(preds.cpu().numpy())
      true_labels.extend(labels.numpy())

  accuracy = accuracy_score(true_labels, predictions)
  print(f"测试准确率: {accuracy:.4f}")


# --- 执行二分类任务 ---
print("=== 二分类任务训练 ===")
binary_model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(device)
bin_optimizer = optim.AdamW(binary_model.parameters(), lr=2e-5)

train_model(binary_model, bin_train_loader, bin_optimizer, device, epochs=5)
evaluate_model(binary_model, bin_test_loader, device)

# --- 执行多分类任务 ---
print("\n=== 多分类任务训练 ===")
multi_model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3
).to(device)
multi_optimizer = optim.AdamW(multi_model.parameters(), lr=2e-5)

train_model(multi_model, multi_train_loader, multi_optimizer, device, epochs=5)
evaluate_model(multi_model, multi_test_loader, device)
