import torch
import torch.nn as nn
from TorchCRF import CRF

class LSTMCRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super(LSTMCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, tags=None):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            loss = -self.crf(emissions, tags, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions)
            return prediction

# 模拟数据生成
vocab_size = 100  # 假设词汇表大小
tagset_size = 5   # 假设标签集大小，例如B、I、O等
model = LSTMCRF(vocab_size, tagset_size)

# 创建输入数据
sentences = torch.randint(0, vocab_size, (4, 10), dtype=torch.long)  # 4个句子，每个句子长度为10
tags = torch.randint(0, tagset_size, (4, 10), dtype=torch.long)       # 真实标签

# 训练模式下计算损失
loss = model(sentences, tags)
print("训练模式下的损失:", loss.item())

# 预测模式下生成序列标签
with torch.no_grad():
    predictions = model(sentences)
print("预测标签序列:", predictions)



