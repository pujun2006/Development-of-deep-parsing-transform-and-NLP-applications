import pandas as pd
import numpy as np
from sklearn.utils import resample
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 构建真实的类别不平衡示例数据 (8个0，2个1 -> 4:1 不平衡)
data = {
    "text": ["样本A", "样本B", "样本C", "样本D", "样本E", "样本F", "样本G", "样本H", "样本I", "样本J"],
    "label": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

print("初始数据分布:\n", df["label"].value_counts())

# --- 策略 A: 下采样（Under-sampling） ---
df_majority = df[df.label == 0]
df_minority = df[df.label == 1]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])
print("\n[下采样后数据分布]:\n", df_balanced["label"].value_counts())

# --- 策略 B: 上采样（Over-sampling） ---
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced_oversample = pd.concat([df_majority, df_minority_upsampled])
print("\n[上采样后数据分布]:\n", df_balanced_oversample["label"].value_counts())

# --- 策略 C: 加权 DataLoader 采样 (WeightedRandomSampler) ---
labels = torch.tensor(df["label"].values, dtype=torch.float)

# 计算每个类别频次与对应权重
num_zeros = (labels == 0).sum().item()
num_ones = (labels == 1).sum().item()

# 样本级权重（样本越少，被抽到的概率越大）
class_weights = [1.0 / num_zeros, 1.0 / num_ones]
sample_weights = torch.tensor([class_weights[int(t)] for t in labels], dtype=torch.double)

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

dataset = TensorDataset(torch.arange(len(labels)), labels)
loader = DataLoader(dataset, sampler=sampler, batch_size=2)


# --- 策略 D: 损失函数加权 (Weighted Loss) ---
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# 核心修正：pos_weight 的正确逻辑是 num_neg / num_pos (即负样本数 / 正样本数)
pos_weight = torch.tensor([num_zeros / num_ones], dtype=torch.float)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练示例（注意：这里演示使用标准无采样 DataLoader 结合加权损失）
standard_loader = DataLoader(dataset, batch_size=2, shuffle=True)

model.train()
for epoch in range(2):
    total_loss = 0.0
    for data_idx, label in standard_loader:
        optimizer.zero_grad()
        output = model(data_idx.float().view(-1, 1))
        loss = criterion(output, label.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(standard_loader):.4f}")