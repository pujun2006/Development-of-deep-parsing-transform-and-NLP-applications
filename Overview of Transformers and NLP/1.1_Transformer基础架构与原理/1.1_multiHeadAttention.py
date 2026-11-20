import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 定义查询、键、值的线性变换矩阵,PyTorch 会在内存中为你自动创建并随机初始化一个权重矩阵
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        # print(self.values.weight.data)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # 批量大小
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 将输入分为多个头
        values = values.reshape(N, value_len, self.heads, self.head_dim) # 3*10*8*8
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算查询和键的点积
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# 测试多头注意力机制的实现
embed_size = 64
heads = 8
seq_length = 10
x = torch.rand((3, seq_length, embed_size))  # 模拟输入数据,3*10*64

mask = None
attention_layer = MultiHeadAttention(embed_size, heads)
output = attention_layer(x, x, x, mask)

# 输出结果
print("多头注意力机制输出形状:", output.shape)
print("多头注意力机制输出:", output)
