import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# 定义情感分析数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 定义BERT情感分类模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        # 冻结部分BERT层
        for param in list(self.bert.parameters())[:int(len(list(self.bert.parameters())) * 0.5)]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        out = self.fc(cls_output)
        return out


# 加载预训练模型和分词器
bert_model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes=2)

# 准备数据
texts = ["这个产品真的很好，使用感受非常棒！", "服务态度差，完全不推荐。", "很满意，物流很快。", "质量不行，很失望。"]
labels = [1, 0, 1, 0]  # 1表示积极，0表示消极

# 划分训练集和测试集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.5, random_state=42)
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# 定义优化器、调度器和损失函数
optimizer = optim.AdamW([
    {'params': model.bert.encoder.layer[-2:].parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 2e-5}
], weight_decay=1e-4)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 0.4]))

# 模拟训练过程中的一个完整epoch，通过3个Epoch反复迭代训练集
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = loss_fn(outputs, batch['label'])
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"Epoch {epoch+1}, 训练损失: {loss.item()}")

# 模型评估
model.eval()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)
        print("真实标签:", labels)
        print("预测结果:", predictions)
