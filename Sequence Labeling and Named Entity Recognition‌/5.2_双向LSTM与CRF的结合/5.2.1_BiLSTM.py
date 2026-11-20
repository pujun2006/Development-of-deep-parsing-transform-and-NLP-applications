import torch
import torch.nn as nn

# 定义双向LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 双向LSTM，设置bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向，因此维度为2倍hidden_size

    def forward(self, x):
        # 初始化LSTM的隐层状态和记忆单元状态，双向则num_layers需*2
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        # LSTM的输出
        out, _ = self.lstm(x, (h0, c0))
        # 仅保留最后时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 参数设置
input_size = 10  # 输入特征维度
hidden_size = 20  # 隐藏层维度
num_layers = 2  # LSTM层数
output_size = 5  # 输出类别数
model = BiLSTMModel(input_size, hidden_size, num_layers, output_size)

# 随机生成输入数据
batch_size = 3
seq_length = 7  # 序列长度
x = torch.randn(batch_size, seq_length, input_size)

# 模型输出
output = model(x)
print("模型输出：")
print(output)
