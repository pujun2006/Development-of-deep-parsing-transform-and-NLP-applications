import torch
import torch.nn as nn
import math


class MultiHeadAttentionWithRegularization(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttentionWithRegularization, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.dense.weight)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(q.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / math.sqrt(dk)
        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        # 保存原始 query 输入（用于正确的残差连接）
        residual = q

        # Query, Key, Value 线性变换
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 分头操作
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 注意力计算
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

        # 恢复维度：transpose 后必须调用 .contiguous() 确保内存连续
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()

        # 拼接多头并通过最终的线性变换
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)

        # 应用 Dropout 和 LayerNorm（正确相加原始输入 residual）
        output = self.dropout(output)
        output = self.layer_norm(output + residual)  # 修复残差连接

        return output, attention_weights


# 设置模型参数
d_model = 512
num_heads = 8
dropout_rate = 0.1
batch_size = 2
seq_len = 10

# 输入张量
input_tensor = torch.rand(batch_size, seq_len, d_model)

# 初始化并测试模型
multi_head_attention = MultiHeadAttentionWithRegularization(d_model, num_heads, dropout_rate)
output, attention_weights = multi_head_attention(input_tensor, input_tensor, input_tensor)

print("Output Shape:", output.shape)
print("Attention Weights Shape:", attention_weights.shape)