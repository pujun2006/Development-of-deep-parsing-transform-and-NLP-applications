import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 数据示例
sentences = ["The book is great!", "The movie was terrible."]
labels = [1, 0]  # 假设1代表积极，0代表消极

# 数据预处理
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# 冻结所有BERT的层
for param in model.bert.parameters():
    param.requires_grad = False

# 解冻特定的层（例如最后两层）
for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

# 定义优化器，仅优化解冻层的参数
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# 训练过程示例
model.train()
for epoch in range(3):  # 假设训练3个周期
    outputs = model(**inputs, labels=torch.tensor(labels))
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1} - Loss: {loss.item()}")

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 示例数据
sentences = ["I love learning about AI.", "Deep learning is fascinating."]
labels = torch.tensor([1, 0])  # 1表示正面，0表示负面

# 预处理数据
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# 冻结所有层的参数
for param in model.bert.parameters():
    param.requires_grad = False

# 定义优化器，仅更新解冻层的参数
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# 逐层解冻的训练过程
model.train()
for epoch in range(3):  # 假设逐层解冻3个周期
    # 每个周期解冻一个新的层
    if epoch < len(model.bert.encoder.layer):
        for param in model.bert.encoder.layer[-(epoch + 1)].parameters():
            param.requires_grad = True

    # 定义损失计算和优化步骤
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 输出每个周期的损失值
    print(f"Epoch {epoch + 1} - Loss: {loss.item()}")

    # 显示当前解冻层
    print(f"Unfrozen Layer: -{epoch + 1}")