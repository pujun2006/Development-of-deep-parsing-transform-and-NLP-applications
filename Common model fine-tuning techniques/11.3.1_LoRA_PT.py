import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim

# 1. 标准 LoRA 线性层实现
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super(LoRALinear, self).__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 冻结原始线性层的权重
        self.linear = original_linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # 定义低秩分解矩阵 A (in -> rank) 和 B (rank -> out)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # 参数初始化：A 用高斯分布，B 用全 0 初始化（确保初始状态增量为 0）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 计算原始权重的输出 (已冻结)
        result = self.linear(x)
        # 2. 计算旁路低秩更新: x @ A.T @ B.T * scaling
        lora_update = (x @ self.lora_A.T) @ self.lora_B.T
        return result + lora_update * self.scaling

import math

# 2. 初始化 BERT 模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 冻结模型的所有基础参数
for param in model.parameters():
    param.requires_grad = False

# 3. 将 LoRA 注入到 BERT 各层的 Self-Attention Query & Value 矩阵中
for layer in model.encoder.layer:
    layer.attention.self.query = LoRALinear(layer.attention.self.query, rank=8, alpha=16.0)
    layer.attention.self.value = LoRALinear(layer.attention.self.value, rank=8, alpha=16.0)

# 4. 只将可训练的 LoRA 参数加入优化器
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=1e-3)

print(f"总训练参数量: {sum(p.numel() for p in trainable_params)} 个")

# 5. 测试前向传播与训练
inputs = tokenizer("LoRA is a parameter-efficient fine-tuning method.", return_tensors="pt")

model.train()
optimizer.zero_grad()
outputs = model(**inputs)
loss = outputs.last_hidden_state.pow(2).mean()
loss.backward()
optimizer.step()

print(f"训练 1 步完成，Loss: {loss.item():.4f}")