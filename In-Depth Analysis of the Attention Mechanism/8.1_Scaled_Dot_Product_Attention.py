import torch
import torch.nn.functional as F

# 设置查询、键和值矩阵的大小
batch_size = 2
seq_len = 4
d_k = 64  # 键和查询的维度
d_v = 64  # 值的维度

# 创建查询、键和值矩阵，初始化为随机数
Q = torch.rand(batch_size, seq_len, d_k)
K = torch.rand(batch_size, seq_len, d_k)
V = torch.rand(batch_size, seq_len, d_v)

# 计算查询和键的点积
scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# 使用softmax归一化，得到注意力权重
attention_weights = F.softmax(scores, dim=-1)

# 使用注意力权重对值矩阵进行加权求和
output = torch.matmul(attention_weights, V)

# 输出结果
print("查询矩阵 Q:\n", Q)
print("键矩阵 K:\n", K)
print("值矩阵 V:\n", V)
print("缩放后的点积 scores:\n", scores)
print("注意力权重 attention_weights:\n", attention_weights)
print("最终输出 output:\n", output)