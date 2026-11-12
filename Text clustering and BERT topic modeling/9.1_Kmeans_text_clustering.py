import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 示例文本数据
documents = [
    "机器学习是一种人工智能技术。",
    "深度学习是机器学习的一个分支。",
    "NLP处理自然语言数据。",
    "监督学习使用标记数据进行训练。",
    "聚类分析是无监督学习的一种。",
    "Python是一种编程语言。",
    "数据科学包括数据分析和统计学。",
    "算法设计是计算机科学的核心。",
    "神经网络是深度学习的基础。",
    "数学在机器学习中很重要。"
]

# 1. 使用 jieba 进行中文分词（用空格连接分词结果）
segmented_documents = [" ".join(jieba.cut(doc)) for doc in documents]

# 2. 使用 TF-IDF 进行文本向量化
# stop_words 可以传入停用词列表（例如："是"、"的"、"在" 等无意义的词）
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(segmented_documents)

# 3. 设置 K 值并进行 K-means 聚类
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X)

# 获取聚类标签和聚类中心
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 输出聚类结果表格
df = pd.DataFrame({'文档': documents, '聚类标签': labels})
print("聚类结果表格:")
print(df.to_string(index=False))

# 计算轮廓系数
silhouette_avg = silhouette_score(X, labels)
print(f"\n聚类的轮廓系数: {silhouette_avg:.4f}")

# 4. 显示每个聚类中心的真·特征关键词
terms = vectorizer.get_feature_names_out()
for i in range(num_clusters):
    print(f"\n聚类 {i} 中心关键词:")
    # 获取特征权重最大的前 5 个词
    cluster_terms = centers[i].argsort()[-5:][::-1]
    for term in cluster_terms:
        print(f" - {terms[term]}")