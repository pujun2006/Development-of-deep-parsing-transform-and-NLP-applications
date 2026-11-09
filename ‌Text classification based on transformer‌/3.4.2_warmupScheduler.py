import torch
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import re


# 自定义数据集
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


# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 数据准备
texts = ["This is a positive example", "This is a negative example"] * 100
labels = [1, 0] * 100

# 加载数据集
train_dataset = SampleDataset(texts, labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 设置优化器和调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 假设3个epoch
warmup_steps = int(0.1 * total_steps)  # 10% 的step用于warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# 训练过程
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

        # 更新学习率
        scheduler.step()

    print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")