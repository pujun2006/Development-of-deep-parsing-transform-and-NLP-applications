import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 定义数据集
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, label_map):
        self.sequences = sequences
        self.labels = labels
        self.label_map = label_map

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = [self.label_map[l] for l in self.labels[idx]]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 样本数据
sequences = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
labels = [["O", "B-ORG", "I-ORG", "O", "O"], ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]]
label_map = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-LOC": 6}

# 加载数据
dataset = SequenceDataset(sequences, labels, label_map)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 定义模型
class SimpleSequenceModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleSequenceModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# 初始化模型与参数
vocab_size = 11
embedding_dim = 8
hidden_dim = 16
output_dim = len(label_map)

model = SimpleSequenceModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# 标签平滑
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# 定义损失和优化器
criterion = LabelSmoothingLoss(smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, output_dim)
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# 测试模型输出
with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(inputs)
        predicted_labels = torch.argmax(outputs, dim=-1)
        print("Predictions:", predicted_labels)
        print("Targets:", targets)
