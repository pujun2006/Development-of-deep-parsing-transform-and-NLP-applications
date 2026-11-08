# 例1-10
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, get_cosine_schedule_with_warmup
import torch.optim as optim

# 定义BERT分类模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        # 冻结部分层
        for param in list(self.bert.parameters())[:int(len(list(self.bert.parameters())) * 0.5)]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        out = self.fc(cls_output)
        return out

# 初始化模型和分词器
bert_model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes=2)

# 定义示例数据
texts = ["这个产品非常好，值得购买。", "产品质量不好，非常失望。"]
encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=32)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# 定义优化器和分层学习率
optimizer = optim.AdamW([
    {'params': model.bert.encoder.layer[-2:].parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 2e-5}
], weight_decay=1e-4)

# 采用余弦退火学习率调度器
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

# 模拟训练过程中的单次前向传播
model.train()
optimizer.zero_grad()
output = model(input_ids, attention_mask)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 0.4]))  # 假设不同类别的损失权重
loss = loss_fn(output, torch.tensor([1, 0]))  # 标签示例
loss.backward()
optimizer.step()
scheduler.step()

# 输出结果
print("BERT模型输出形状:", output.shape)
print("BERT模型输出:", output)
print("训练损失:", loss.item())
