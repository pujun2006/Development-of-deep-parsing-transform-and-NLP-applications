import os
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# 1. 设置国内镜像源（防止下载 BERT 权重超时报错）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 初始化 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 输入句子
text = "The quick brown fox jumps over the lazy dog"
tokens = tokenizer(text, return_tensors='pt', padding=True)


# 2. 定义规范的 GNN 层
class GNNLayer(nn.Module):

  def __init__(self, in_features, out_features):
    super(GNNLayer, self).__init__()
    self.linear = nn.Linear(in_features, out_features)

  def forward(self, node_features, adj_matrix):
    # 消息聚合：(N, N) * (N, in_features) -> (N, in_features)
    aggregated = torch.matmul(adj_matrix, node_features)
    return torch.relu(self.linear(aggregated))


# 3. 规范的 Bert + GNN 混合模型
class BertGNNModel(nn.Module):

  def __init__(self, bert_model, gnn_layer):
    super(BertGNNModel, self).__init__()
    self.bert = bert_model
    self.gnn = gnn_layer

  def forward(self, input_ids, attention_mask, adj_matrix):
    # 动态获取 BERT 输出特征: (1, seq_len, 768)
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)

    # 通过 GNN 更新节点特征: (seq_len, 128)
    gnn_output = self.gnn(token_embeddings, adj_matrix)
    return gnn_output


# 4. 动态构建匹配 Token 长度的邻接矩阵
seq_len = tokens['input_ids'].shape[1]  # 自动获取实际 Token 数量 (11)

G = nx.DiGraph()
# 自动构造覆盖所有 Token (0 到 seq_len-1) 的链式边
edges = [(i, i + 1) for i in range(seq_len - 1)]
G.add_edges_from(edges)

# 生成 (11, 11) 的邻接矩阵
adj_matrix = nx.adjacency_matrix(G).todense()
adj_matrix = torch.FloatTensor(adj_matrix)

# 添加自环（Self-loop）：让节点在聚合邻居信息时，保留自身的 BERT 语义
adj_matrix += torch.eye(seq_len)

# 5. 初始化模型并前向传播
gnn_layer = GNNLayer(in_features=768, out_features=128)
model = BertGNNModel(bert_model, gnn_layer)

gnn_output = model(
    input_ids=tokens['input_ids'],
    attention_mask=tokens['attention_mask'],
    adj_matrix=adj_matrix,
)

# 输出验证
print(f"Token 实际数量 (节点数): {seq_len}")
print(f"邻接矩阵形状: {adj_matrix.shape}")
print(f"混合模型输出的特征矩阵形状: {gnn_output.shape}")  # torch.Size([11, 128])
print("\n输出的部分节点特征：\n", gnn_output[:3, :5])  # 查看前3个节点的前5维