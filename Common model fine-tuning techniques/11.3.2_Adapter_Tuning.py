import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

# 1. 加载模型与分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 2. 定义标准的 Bottleneck Adapter 模块
class Adapter(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=64):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()  # BERT 通常搭配 GELU 激活函数
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        adapter_out = self.up_project(self.activation(self.down_project(x)))
        return self.layer_norm(residual + adapter_out)

# 3. 包装完整的 Output 模块，将 Adapter 真正融入前向传播 (Forward)
class AdapterOutputWrapper(nn.Module):
    def __init__(self, original_output, bottleneck_dim=64):
        super(AdapterOutputWrapper, self).__init__()
        self.original_output = original_output
        self.adapter = Adapter(original_output.dense.out_features, bottleneck_dim)

    def forward(self, hidden_states, input_tensor):
        # 先执行原本 BERT Output 层的 Dense + Dropout + LayerNorm
        orig_out = self.original_output(hidden_states, input_tensor)
        # 再将结果送入 Adapter 进行残差学习
        return self.adapter(orig_out)

# 4. 将 Adapter 注入到 BERT 各层的 Self-Attention Output 和 FFN Output 之后
for layer in model.encoder.layer:
    # 替换 Attention 后的 Output 层
    layer.attention.output = AdapterOutputWrapper(layer.attention.output)
    # 替换 FFN 后的 Output 层
    layer.output = AdapterOutputWrapper(layer.output)

# 5. 冻结 BERT 原生参数，仅解冻 Adapter 参数
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "adapter" in name:
        param.requires_grad = True

# 6. 正确过滤出可训练参数并初始化优化器
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=1e-3)

print(f"可训练 Adapter 参数量: {sum(p.numel() for p in trainable_params)} / 总参数量: {sum(p.numel() for p in model.parameters())}")

# 7. 模型训练与验证
text = "Adapter Tuning is a method for efficient model fine-tuning."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

model.train()
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = outputs.last_hidden_state.pow(2).mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")