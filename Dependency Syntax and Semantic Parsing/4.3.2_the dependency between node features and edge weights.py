import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# 定义依存图模型，考虑边权重
class WeightedDependencyGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WeightedDependencyGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

# 构建句法依存关系图，并定义边权重
G = nx.DiGraph()
edges = [
    (0, 1, 0.5), (1, 2, 0.7), (2, 3, 0.8), (3, 4, 0.6),  # 主谓宾关系
    (4, 5, 0.4), (5, 6, 0.9), (6, 7, 0.3), (3, 8, 0.5)   # 修饰关系
]
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# 初始化节点特征，示例为9个节点，特征维度为3
node_features = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1],
                              [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1],
                              [1, 1, 0]], dtype=torch.float)
graph_data = from_networkx(G)
graph_data.x = node_features
edge_weights = torch.tensor([edge[2] for edge in edges], dtype=torch.float)
graph_data.edge_weight = edge_weights

# 定义模型参数
input_dim = 3
hidden_dim = 8
output_dim = 4
model = WeightedDependencyGNN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 模拟目标输出用于损失计算
target = torch.rand((graph_data.num_nodes, output_dim))

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# 测试模型输出
model.eval()
output = model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
print("依存关系节点嵌入表示：")
print(output)



