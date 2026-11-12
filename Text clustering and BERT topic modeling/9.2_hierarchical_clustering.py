import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram

# 修复核心报错：在导入 pyplot 前显式设置后端
import matplotlib
matplotlib.use('TkAgg')  # 如果在无界面服务器运行，可更改为 'Agg' 并在末尾保存图片
import matplotlib.pyplot as plt

# 示例文本数据集
texts = [
    "Machine learning is fascinating.",
    "Artificial intelligence and machine learning are closely related.",
    "Clustering algorithms help in grouping data.",
    "Natural language processing and AI have great potential.",
    "Hierarchical clustering builds a tree structure of clusters.",
    "Text clustering is useful for organizing information.",
    "AI and machine learning are advancing rapidly.",
    "Data analysis is essential for insights."
]

# 1. 文本数据向量化
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts).toarray()

# 2 & 3. 使用层次聚类（修正：直接将特征矩阵与 cosine 距离度量结合使用）
# 在 cosine 距离下使用 'average' 或 'complete' 联动算法
linkage_matrix = linkage(tfidf_matrix, method='average', metric='cosine')

# 4. 绘制层次聚类树状图
plt.figure(figsize=(12, 7))
dendrogram(
    linkage_matrix,
    labels=texts,
    orientation='left',  # 改为左侧排列，文本显示更清晰
    leaf_font_size=10
)
plt.title('Hierarchical Clustering Dendrogram (Cosine Distance)')
plt.xlabel('Distance')
plt.ylabel('Texts')
plt.tight_layout()
plt.show()