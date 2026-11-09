import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import re


# 数据集类定义
class SampleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = re.sub(r"[^\w\s]", "", self.texts[idx]).lower()
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 冻结前几层参数
for name, param in model.named_parameters():
    if "encoder.layer.0" in name or "encoder.layer.1" in name or "encoder.layer.2" in name:
        param.requires_grad = False

# 模拟数据
texts = ["This is a positive example", "This is a negative example"] * 100
labels = [1, 0] * 100

# 数据加载
train_dataset = SampleDataset(texts, labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

# 训练代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(3):  # 假设训练3个epoch
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # 更新权重
        optimizer.step()

    # 每个epoch结束时更新学习率
    scheduler.step()
    print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")