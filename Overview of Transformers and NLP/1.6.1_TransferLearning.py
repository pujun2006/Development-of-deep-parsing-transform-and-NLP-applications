import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import torch.optim as optim

# 定义BERT分类模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

        # 冻结BERT前几层
        for param in list(self.bert.parameters())[:int(len(list(self.bert.parameters())) * 0.7)]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        out = self.fc(cls_output)
        return out

# 初始化分词器和模型
bert_model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes=2)

# 定义数据和优化器
texts = ["这是一个积极的例子。", "这是一个消极的例子。"]
encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=32)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# 优化器和学习率调度器
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

# 训练循环中的单次前向传播示例
model.train()
optimizer.zero_grad()
output = model(input_ids, attention_mask)
loss = nn.CrossEntropyLoss()(output, torch.tensor([1, 0]))  # 假设标签为[1, 0]
loss.backward()
optimizer.step()
scheduler.step()

# 输出结果
print("BERT模型输出形状:", output.shape)
print("BERT模型输出:", output)
print("训练损失:", loss.item())
