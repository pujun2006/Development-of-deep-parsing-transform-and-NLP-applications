import torch
import torch.nn as nn

# 定义ELMo模型
class ELMo(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ELMo, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 多层双向LSTM
        self.lstm_layers = nn.ModuleList(
            [nn.LSTM(input_size if i == 0 else hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
             for i in range(num_layers)]
        )
        # 最后线性层，用于序列标注
        self.classifier = nn.Linear(hidden_size * 2 * num_layers, 3)  # 假设3类序列标签

    def forward(self, x):
        hidden_states = []
        for lstm in self.lstm_layers:
            h0 = torch.zeros(2, x.size(0), self.hidden_size)  # 双向LSTM需2倍的hidden_size
            c0 = torch.zeros(2, x.size(0), self.hidden_size)
            x, _ = lstm(x, (h0, c0))
            hidden_states.append(x)

        # 拼接每层LSTM的输出，形成最终的ELMo嵌入
        elmo_embedding = torch.cat(hidden_states, dim=2)
        # 通过分类层进行标签预测
        output = self.classifier(elmo_embedding)
        return output

# 参数设置
input_size = 10  # 输入特征维度
hidden_size = 20  # LSTM隐藏层维度
num_layers = 2  # LSTM层数
elmo_model = ELMo(input_size, hidden_size, num_layers)

# 生成模拟输入数据
batch_size = 2
seq_length = 5  # 序列长度
x = torch.randn(batch_size, seq_length, input_size)

# 模型输出
output = elmo_model(x)
print("模型输出：")
print(output)



