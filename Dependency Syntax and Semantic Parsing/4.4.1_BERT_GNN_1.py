import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# 1. 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 2. 输入文本并生成BERT嵌入
text = 'The cat sat on the mat'
tokens = tokenizer(text, return_tensors='pt', padding=True)
with torch.no_grad():
  bert_outputs = bert_model(**tokens)

# 提取特征：形状为 (seq_len, hidden_size) -> (8, 768)
token_embeddings = bert_outputs.last_hidden_state.squeeze(0)
num_nodes = token_embeddings.shape[0]  # 动态获取节点数量 (8)


# 3. 定义简单图神经网络层
class GNNLayer(nn.Module):

  def __init__(self, in_features, out_features):
    super(GNNLayer, self).__init__()
    self.linear = nn.Linear(in_features, out_features)

  def forward(self, node_features, adj_matrix):
    # 消息传递：邻接矩阵 * 节点特征
    aggregate_features = torch.matmul(adj_matrix, node_features)
    return torch.relu(self.linear(aggregate_features))


# 4. 构建匹配实际 Token 数量的依存关系图 (8个节点: 0到7)
G = nx.DiGraph()

# 示例：构造一个覆盖 8 个 Token 的简单链式结构，并添加自环 (Self-loops)
edges = [(i, i + 1) for i in range(num_nodes - 1)]
G.add_edges_from(edges)

# 生成邻接矩阵 (8x8)
adj_matrix = nx.adjacency_matrix(G).todense()
adj_matrix = torch.FloatTensor(adj_matrix)

# 添加自环（让每个节点在聚合邻居信息的同时，保留自己的特征）
adj_matrix += torch.eye(num_nodes)

# 5. 定义并运行 GNN
gnn_layer = GNNLayer(token_embeddings.shape[1], 128)
gnn_output = gnn_layer(token_embeddings, adj_matrix)

# 输出结果验证
print(f'Token 数量 (节点数): {num_nodes}')
print(f'邻接矩阵形状: {adj_matrix.shape}')
print(f'更新后的节点特征矩阵形状：{gnn_output.shape}')  # 应为 (8, 128)
print('更新后的节点特征矩阵：')
print(gnn_output)