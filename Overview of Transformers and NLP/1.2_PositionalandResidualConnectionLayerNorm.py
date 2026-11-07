import torch
import torch.nn as nn
import math


# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size

        # 创建一个位置编码矩阵，大小为 (max_len, embed_size)
        position_encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        # 奇偶维度分别使用sin和cos进行编码
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度并设置为不可训练,.detach() 从计算图中剥离，确保位置编码不参与梯度更新（不可训练）
        self.position_encoding = position_encoding.unsqueeze(0).detach()

    def forward(self, x):
        # 将位置编码添加到输入张量中
        x = x * math.sqrt(self.embed_size)  # 缩放
        seq_len = x.size(1)
        x = x + self.position_encoding[:, :seq_len, :].to(x.device)
        return x




# 定义残差连接与层归一化模块
class ResidualConnectionLayerNorm(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super(ResidualConnectionLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        # 残差连接与层归一化
        return self.norm(x + self.dropout(sublayer_output))


def main():
  # 在这里编写你的核心逻辑或调用其他函数
  print("程序开始运行...")
  # 测试位置编码实现
  embed_size = 64
  seq_length = 10
  x = torch.zeros((3, seq_length, embed_size))  # 输入为零张量

  position_encoding = PositionalEncoding(embed_size)
  output = position_encoding(x)

  print("位置编码后的输出形状:", output.shape)
  print("位置编码后的输出:", output)
  # 测试残差连接与层归一化模块
  residual_layer = ResidualConnectionLayerNorm(embed_size)
  sublayer_output = torch.rand((3, seq_length, embed_size))  # 模拟子层输出
  residual_output = residual_layer(output, sublayer_output)

  # 输出结果
  print("残差连接与层归一化输出形状:", residual_output.shape)


if __name__ == "__main__":
  main()
