import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. 模拟数据并进行 L2 归一化（跨语言对齐的必备步骤）
np.random.seed(42)
source_embeddings = np.random.randn(100, 300)  # 使用正态分布
target_embeddings = np.random.randn(100, 300)

# L2 范数归一化
source_embeddings = source_embeddings / np.linalg.norm(
    source_embeddings, axis=1, keepdims=True
)
target_embeddings = target_embeddings / np.linalg.norm(
    target_embeddings, axis=1, keepdims=True
)

# 2. 对齐前的平均余弦相似度
initial_sim = np.diag(
    cosine_similarity(source_embeddings, target_embeddings)
).mean()
print(f"Initial Pairwise Similarity (Before Alignment): {initial_sim:.4f}")


# 3. 使用正交 Procrustes 算法训练投影矩阵 W (正规闭式解)
def train_orthogonal_projection(src_emb, tgt_emb):
  # 计算跨语言协方差矩阵 M = X^T * Y
  M = src_emb.T.dot(tgt_emb)

  # 奇异值分解 SVD
  U, S, Vh = np.linalg.svd(M)

  # 计算最佳正交映射矩阵 W = U * V^T
  projection_matrix = U.dot(Vh)
  return projection_matrix


# 训练投影矩阵
W = train_orthogonal_projection(source_embeddings, target_embeddings)

# 4. 验证投影矩阵的正交性 (W^T * W 应等于单位矩阵 I)
is_orthogonal = np.allclose(W.T.dot(W), np.eye(300))
print(f"Projection Matrix is Orthogonal: {is_orthogonal}")

# 5. 使用投影矩阵对齐源语言嵌入
projected_src = source_embeddings.dot(W)

# 6. 计算对齐后的词对平均余弦相似度 (主对角线上的相似度)
aligned_sim = np.diag(cosine_similarity(projected_src, target_embeddings)).mean()
print(f"Aligned Pairwise Similarity (After Alignment):  {aligned_sim:.4f}")