import torch
import torch.nn as nn


class TreeLSTMCellWithScoring(nn.Module):

  def __init__(self, input_dim, hidden_dim):
    super(TreeLSTMCellWithScoring, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    # 门控定义
    self.W_i = nn.Linear(input_dim, hidden_dim)
    self.U_i = nn.Linear(hidden_dim, hidden_dim)

    self.W_f = nn.Linear(input_dim, hidden_dim)
    self.U_f = nn.Linear(hidden_dim, hidden_dim)

    self.W_o = nn.Linear(input_dim, hidden_dim)
    self.U_o = nn.Linear(hidden_dim, hidden_dim)

    self.W_u = nn.Linear(input_dim, hidden_dim)
    self.U_u = nn.Linear(hidden_dim, hidden_dim)

    # 打分层
    self.scoring_layer = nn.Linear(hidden_dim, 1)

  def forward(self, x, child_h, child_c):
    # child_h 形状: (num_children, 1, hidden_dim)
    child_h_sum = torch.sum(child_h, dim=0)

    i = torch.sigmoid(self.W_i(x) + self.U_i(child_h_sum))
    o = torch.sigmoid(self.W_o(x) + self.U_o(child_h_sum))
    u = torch.tanh(self.W_u(x) + self.U_u(child_h_sum))

    f_list = []
    for h in child_h:
      f = torch.sigmoid(self.W_f(x) + self.U_f(h))
      f_list.append(f)

    f_c_sum = torch.sum(
        torch.stack([f * c for f, c in zip(f_list, child_c)]), dim=0
    )

    c = i * u + f_c_sum
    h = o * torch.tanh(c)

    # 计算打分值
    score = self.scoring_layer(h)

    return h, c, score


class TreeLSTMWithScoring(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    super(TreeLSTMWithScoring, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.cell = TreeLSTMCellWithScoring(embedding_dim, hidden_dim)
    self.hidden_dim = hidden_dim

  def forward(self, tree):
    # 简化接口，直接从树根节点遍历
    h, c, score = self.recursive_forward(tree)
    return h, score

  def recursive_forward(self, node):
    # 1. 修复：将 int 转换为 LongTensor，形状为 (1, embedding_dim)
    node_idx_tensor = torch.tensor([node.word_idx], dtype=torch.long)
    x = self.embedding(node_idx_tensor)

    if node.is_leaf():
      # 2. 修复：叶节点的子节点伪隐状态设为 (1, 1, hidden_dim)
      child_h = torch.zeros(1, 1, self.hidden_dim)
      child_c = torch.zeros(1, 1, self.hidden_dim)
    else:
      child_h_list, child_c_list = [], []
      for child in node.children:
        # 3. 修复：一致接收 recursive_forward 返回的 3 个值 (h, c, score)
        h, c, score = self.recursive_forward(child)
        child_h_list.append(h)
        child_c_list.append(c)

      # 4. 修复：统一堆叠维度为 (num_children, 1, hidden_dim)
      child_h = torch.stack(child_h_list, dim=0).squeeze(1)
      child_c = torch.stack(child_c_list, dim=0).squeeze(1)

    # 统一调用 Cell
    h, c, score = self.cell(x, child_h, child_c)

    # 5. 修复：正确返回 h, c, score 三元组以供上一层递归解构
    return h, c, score


class TreeNode:

  def __init__(self, word_idx):
    self.word_idx = word_idx
    self.children = []

  def add_child(self, child):
    self.children.append(child)

  def is_leaf(self):
    return len(self.children) == 0


# 参数定义
vocab_size = 100
embedding_dim = 50
hidden_dim = 100

# 构建树结构
root = TreeNode(0)
child1 = TreeNode(1)
child2 = TreeNode(2)
child3 = TreeNode(3)
root.add_child(child1)
root.add_child(child2)
child1.add_child(child3)

# 运行模型
model = TreeLSTMWithScoring(vocab_size, embedding_dim, hidden_dim)
output, score = model(root)

print("Tree-LSTM 输出向量形状:", output.shape)  # torch.Size([1, 100])
print("依存打分值 (Score) 形状:", score.shape)  # torch.Size([1, 1])
print("打分数值:", score.item())