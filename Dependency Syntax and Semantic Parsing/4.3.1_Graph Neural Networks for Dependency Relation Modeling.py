import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# 定义GNN模型，使用两层GCNConv实现依存信息传播
class DependencyGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DependencyGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 构建句法依存关系图，示例句为："The quick brown fox jumps over the lazy dog"
G = nx.DiGraph()
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 主谓宾结构
    (4, 5), (5, 6), (6, 7), (3, 8)   # 修饰关系
]
for u, v in edges:
    G.add_edge(u, v)

# 初始化节点特征，8个单词每个节点用3维特征表示
node_features = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1],
                              [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1],
                              [1, 1, 0]], dtype=torch.float)
graph_data = from_networkx(G)
graph_data.x = node_features

# 定义模型参数
input_dim = 3
hidden_dim = 8
output_dim = 4
model = DependencyGNN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 模拟目标输出用于计算损失
target = torch.rand((graph_data.num_nodes, output_dim))

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# 测试模型的依存关系嵌入
model.eval()
output = model(graph_data.x, graph_data.edge_index)
print("依存关系节点嵌入表示：")
print(output)


