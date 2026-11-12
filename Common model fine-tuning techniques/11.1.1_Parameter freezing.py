import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# 1. 加载预训练模型与分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 2. 示例数据
sentences = ["Machine learning is evolving.", "Natural language processing is intriguing."]
labels = torch.tensor([1, 0])

# 3. 预处理数据
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# 4. 精准冻结参数：冻结 Embeddings 和 Encoder 的前 10 层 (layer 0 ~ 9)
# 允许训练的层包括：layer.10, layer.11, pooler, classifier
for name, param in model.named_parameters():
    # 如果参数属于前 10 层 Transformer 或者是 Embeddings 层，则冻结
    if "embeddings" in name or any(f"encoder.layer.{i}." in name for i in range(10)):
        param.requires_grad = False
    else:
        param.requires_grad = True

# 检查并打印将被训练的层结构
trainable_layers = [name for name, param in model.named_parameters() if param.requires_grad]
print("=== 可训练的层列表 ===")
for layer in trainable_layers:
    print(layer)
print("====================\n")

# 5. 定义优化器（仅对 requires_grad=True 的参数进行更新）
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# 6. 微调训练过程
model.train()
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}")