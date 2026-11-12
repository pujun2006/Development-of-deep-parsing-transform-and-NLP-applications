from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载Sentence-BERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 定义示例文本
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A fast, dark-colored fox leaps over a sleepy canine",
    "The sun rises in the east and sets in the west",
    "Sunshine comes from the east while dusk settles in the west",
    "Artificial intelligence is transforming industries",
    "Machine learning and AI are revolutionizing various fields"
]

# 生成嵌入向量
embeddings = model.encode(texts)

# 打印嵌入的形状和一个示例嵌入
print("Embedding shape:", embeddings.shape)
print("Sample embedding for first sentence:\n", embeddings[0])

# 计算嵌入的余弦相似度
similarity_matrix = cosine_similarity(embeddings)
print("Cosine Similarity Matrix:\n", similarity_matrix)
