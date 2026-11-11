import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# Step 1: 模拟数据并进行 L2 归一化 (跨语言对齐的必备步骤)
source_embeddings = np.random.rand(5000, 300)  # 5000 个英文词，300 维
target_embeddings = np.random.rand(5000, 300)  # 5000 个法文词，300 维

# L2 归一化：将词向量模长化为 1
source_embeddings = source_embeddings / np.linalg.norm(
    source_embeddings, axis=1, keepdims=True
)
target_embeddings = target_embeddings / np.linalg.norm(
    target_embeddings, axis=1, keepdims=True
)

source_embeddings = torch.FloatTensor(source_embeddings)
target_embeddings = torch.FloatTensor(target_embeddings)


# Step 2: 定义生成器（映射矩阵）与判别器
class Generator(nn.Module):

  def __init__(self, input_dim):
    super(Generator, self).__init__()
    # 映射矩阵 W (无 bias)
    self.linear = nn.Linear(input_dim, input_dim, bias=False)
    # 初始化为单位矩阵 I，加速收敛
    nn.init.eye_(self.linear.weight)

  def forward(self, x):
    return self.linear(x)


class Discriminator(nn.Module):

  def __init__(self, input_dim):
    super(Discriminator, self).__init__()
    # 移除最后一层的 Sigmoid，配合 BCEWithLogitsLoss 使用 (防 log(0) NaN)
    self.model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.LeakyReLU(0.2),  # 使用 LeakyReLU 替代 standard ReLU 防止神经元死亡
        nn.Linear(128, 64),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 1),
    )

  def forward(self, x):
    return self.model(x)


# 初始化模型
input_dim = 300
generator = Generator(input_dim)
discriminator = Discriminator(input_dim)

# 使用数值稳定的 BCEWithLogitsLoss (包含内置 Sigmoid)
criterion = nn.BCEWithLogitsLoss()

# 适当降低学习率，提升 GAN 训练稳定性
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# Step 3: 对抗训练过程
epochs = 1000
batch_size = 128

print("开始对抗训练...")
for epoch in range(epochs):
  # 采样 Batch
  idx_src = np.random.choice(5000, batch_size, replace=False)
  idx_tgt = np.random.choice(5000, batch_size, replace=False)

  source_batch = source_embeddings[idx_src]
  target_batch = target_embeddings[idx_tgt]

  # ---------------------
  #  1. 训练判别器 D
  # ---------------------
  d_optimizer.zero_grad()

  fake_target = generator(source_batch)

  # 标签：真实为 1，伪造为 0
  real_labels = torch.ones(batch_size, 1)
  fake_labels = torch.zeros(batch_size, 1)

  real_loss = criterion(discriminator(target_batch), real_labels)
  fake_loss = criterion(discriminator(fake_target.detach()), fake_labels)

  d_loss = (real_loss + fake_loss) / 2
  d_loss.backward()

  # 梯度裁剪：防止判别器梯度爆炸导致的数值不稳定性
  torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
  d_optimizer.step()

  # ---------------------
  #  2. 训练生成器 G
  # ---------------------
  g_optimizer.zero_grad()

  fake_target = generator(source_batch)
  # 生成器的目标是骗过判别器 (让判别器认为是 1)
  g_loss = criterion(discriminator(fake_target), real_labels)

  g_loss.backward()

  # 梯度裁剪：防止生成器梯度爆炸
  torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
  g_optimizer.step()

  # ---------------------
  #  3. 正交步骤 (Safe Orthogonalization)
  # ---------------------
  with torch.no_grad():
    W = generator.linear.weight.data

    # 防御机制 1：如果存在 NaN 或 Inf 则跳过正交化
    if not torch.isnan(W).any() and not torch.isinf(W).any():
      try:
        # 使用现代 PyTorch 建议的 torch.linalg.svd 接口
        # 注意：torch.linalg.svd 返回的是 (U, S, Vh)，其中 Vh 是 V 的转置 (V^T)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # 更新正交矩阵：W = U * V^T (即 U * Vh)
        generator.linear.weight.data = torch.mm(U, Vh)
      except Exception as e:
        # 防御机制 2：捕捉由于病态矩阵引发的 LinAlgError 异常，保障训练不崩溃
        pass

  # 打印进度
  if (epoch + 1) % 100 == 0:
    print(
        f"Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss:"
        f" {g_loss.item():.4f}"
    )

# Step 4: 导出对齐后的嵌入并计算相似度
aligned_source_embeddings = generator(source_embeddings).detach().numpy()
target_embeddings_np = target_embeddings.numpy()

similarities = cosine_similarity(
    aligned_source_embeddings, target_embeddings_np
)
print("\n对齐后的前 5 个词对相似度矩阵：")
print(similarities[:5, :5])