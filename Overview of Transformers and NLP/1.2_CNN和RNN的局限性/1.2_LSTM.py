import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义LSTM模型
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # 嵌入层
        out, (h_n, c_n) = self.lstm(x)  # LSTM层
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)  # 全连接层
        return out

# 测试TextLSTM模型
vocab_size = 5000
embed_size = 128
hidden_size = 256
num_layers = 2
num_classes = 2
seq_length = 100

model = TextLSTM(vocab_size, embed_size, hidden_size, num_layers, num_classes)
sample_input = torch.randint(0, vocab_size, (32, seq_length))  # 模拟输入数据
output = model(sample_input)

# 输出模型结果
print("TextLSTM模型输出形状:", output.shape)
print("TextLSTM模型输出:", output)
