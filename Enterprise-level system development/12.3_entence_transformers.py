from sentence_transformers import SentenceTransformer
import numpy as np
import jieba
from sklearn.decomposition import PCA

# 1. 初始化 SBERT 模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 样本数据
texts = [
    "这款产品非常好用，功能强大且易操作。",
    "服务态度差，体验非常糟糕。",
    "产品质量一般，但价格实惠。",
    "这是我用过的最好的一款应用。",
    "这家餐厅的服务真的很差劲。",
    "这件商品的性价比非常高，值得推荐！"
]

# 生成 384 维文本嵌入
embeddings = model.encode(texts)

# 2. 示例情感词典
sentiment_dict = {"好用": 1, "强大": 1, "差劲": -1, "糟糕": -1, "推荐": 1, "好": 1}


# 情感增强函数（修正：使用 jieba 中文切词）
def enhance_embedding(text, embedding):
    words = jieba.lcut(text)  # 正确切出中文词语
    weights = [sentiment_dict.get(word, 0) for word in words]

    # 防止无匹配词时 np.mean 报 RuntimeWarning，无词匹配则设增强因子为 0
    enhancement_factor = np.mean(weights) if len(weights) > 0 else 0
    return embedding * (1 + enhancement_factor)


# 应用情感增强
enhanced_embeddings = np.array([enhance_embedding(text, emb) for text, emb in zip(texts, embeddings)])

# 3. PCA 降维（修正：由于 n_samples=6，n_components 必须 <= 6，此处降到 2 维）
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(enhanced_embeddings)

# 输出降维结果
for i, reduced_embedding in enumerate(reduced_embeddings):
    print(f"Text: {texts[i]}")
    print(f"Reduced Embedding (2D): {reduced_embedding}\n")

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import pandas as pd

# 示例文本数据和情感分类结果
texts = [
    "产品非常好用，强烈推荐",
    "非常不满意，质量差",
    "服务不错，值得推荐",
    "性价比很高，非常划算",
    "很失望，不会再购买"
]

# 使用Sentence-BERT模型生成文本嵌入向量
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(texts)

# 查看生成的嵌入向量
for i, embedding in enumerate(embeddings):
    print(f"Text: {texts[i]}")
    print(f"Embedding: {embedding[:5]}... (length: {len(embedding)})\n")

# 定义聚类数（例如，分为两类：正面、负面）
num_clusters = 2

# 初始化KMeans模型并进行聚类
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(embeddings)

# 获取每个文本的聚类标签
cluster_labels = kmeans.labels_

# 打印每条文本的聚类结果
for i, label in enumerate(cluster_labels):
    print(f"Text: {texts[i]} -> Cluster Label: {label}")
