import torch
import torch.nn as nn


class TreeLSTMCell(nn.Module):

  def __init__(self, input_dim, hidden_dim):
    super(TreeLSTMCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    # 输入门、遗忘门、输出门和候选记忆单元的线性层定义
    self.W_i = nn.Linear(input_dim, hidden_dim)
    self.U_i = nn.Linear(hidden_dim, hidden_dim)

    self.W_f = nn.Linear(input_dim, hidden_dim)
    self.U_f = nn.Linear(hidden_dim, hidden_dim)

    self.W_o = nn.Linear(input_dim, hidden_dim)
    self.U_o = nn.Linear(hidden_dim, hidden_dim)

    self.W_u = nn.Linear(input_dim, hidden_dim)
    self.U_u = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, x, child_h, child_c):
    # child_h: (num_children, 1, hidden_dim)
    # child_h_sum: (1, hidden_dim)
    child_h_sum = torch.sum(child_h, dim=0)

    # 计算输入门、输出门和候选记忆单元
    i = torch.sigmoid(self.W_i(x) + self.U_i(child_h_sum))
    o = torch.sigmoid(self.W_o(x) + self.U_o(child_h_sum))
    u = torch.tanh(self.W_u(x) + self.U_u(child_h_sum))

    # 计算每一个子节点的遗忘门并累加子节点的记忆单元
    f_list = []
    for h in child_h:
      f = torch.sigmoid(self.W_f(x) + self.U_f(h))
      f_list.append(f)

    # 累加 f * c
    f_c_sum = torch.sum(
        torch.stack([f * c for f, c in zip(f_list, child_c)]), dim=0
    )

    # 更新记忆单元 c 和隐状态 h
    c = i * u + f_c_sum
    h = o * torch.tanh(c)

    return h, c


class TreeLSTM(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    super(TreeLSTM, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.cell = TreeLSTMCell(embedding_dim, hidden_dim)
    self.hidden_dim = hidden_dim

  def forward(self, tree):
    h, c = self.recursive_forward(tree)
    return h

  def recursive_forward(self, node):
    # 核心修复 1：将 int 转换为 1D LongTensor 并获取 embedding, 形状为 (1, embedding_dim)
    node_idx_tensor = torch.tensor([node.word_idx], dtype=torch.long)
    x = self.embedding(node_idx_tensor)

    if node.is_leaf():
      # 核心修复 2：叶节点的子节点隐藏状态初始化为 (1, 1, hidden_dim)
      child_h = torch.zeros(1, 1, self.hidden_dim)
      child_c = torch.zeros(1, 1, self.hidden_dim)
    else:
      # 递归计算各个子节点的隐状态
      child_h_list, child_c_list = [], []
      for child in node.children:
        h, c = self.recursive_forward(child)
        child_h_list.append(h)
        child_c_list.append(c)

      # 核心修复 3：堆叠形成 3D 张量 (num_children, 1, hidden_dim)
      child_h = torch.stack(child_h_list, dim=0).squeeze(1)
      child_c = torch.stack(child_c_list, dim=0).squeeze(1)

    return self.cell(x, child_h, child_c)


class TreeNode:

  def __init__(self, word_idx):
    self.word_idx = word_idx
    self.children = []

  def add_child(self, child):
    self.children.append(child)

  def is_leaf(self):
    return len(self.children) == 0


# 模型参数
vocab_size = 100
embedding_dim = 50
hidden_dim = 100

# 构建语法树结构
root = TreeNode(0)
child1 = TreeNode(1)
child2 = TreeNode(2)
child3 = TreeNode(3)

root.add_child(child1)
root.add_child(child2)
child1.add_child(child3)

# 模型初始化与前向传播
model = TreeLSTM(vocab_size, embedding_dim, hidden_dim)
output = model(root)

print("根节点最终输出向量形状:", output.shape)  # 输出应为 torch.Size([1, 100])
print("部分输出值:", output[0, :5])

