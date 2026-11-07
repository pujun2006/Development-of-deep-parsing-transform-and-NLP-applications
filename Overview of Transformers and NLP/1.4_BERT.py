# 例1-7
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


# 定义BERT分类模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        # BERT模型输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 取[CLS]标记对应的隐藏状态
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Dropout和全连接层进行分类
        cls_output = self.dropout(cls_output)
        out = self.fc(cls_output)
        return out


# 加载BERT分词器和模型
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes=2)

# 输入文本示例
texts = ["This is a positive example.", "This is a negative example."]
encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=32)

# 模型推理
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
output = model(input_ids, attention_mask)

# 输出结果
print("BERT模型输出形状:", output.shape)
print("BERT模型输出:", output)
print("转换为概率：",torch.softmax(output, dim=1))
