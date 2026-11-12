import torch
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 核心修复：在导入 pyplot 之前显式切换后端，解决 PyCharm 插件后端冲突
import matplotlib
matplotlib.use('TkAgg')  # 切换为标准 GUI 后端
import matplotlib.pyplot as plt

# Step 1: 加载 BERT 模型和 Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# 示例文本数据
texts = [
    "Machine learning allows computers to learn from data.",
    "Natural language processing makes human language accessible to machines.",
    "Renewable energy sources are crucial for a sustainable future.",
    "Artificial intelligence is transforming the healthcare industry.",
    "Climate change impacts biodiversity and ecosystems worldwide."
]

# Step 2: 使用 BERT 生成文本嵌入
def get_bert_embeddings(texts):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embedding.squeeze())
    return np.array(embeddings)

embeddings = get_bert_embeddings(texts)

# Step 3: 使用 CountVectorizer 和 LDA 进行主题建模
vectorizer = CountVectorizer(stop_words='english')
text_vectors = vectorizer.fit_transform(texts)
lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
lda_topics = lda_model.fit_transform(text_vectors)

# Step 4: 将 BERT 嵌入通过 PCA 降维至 2D 空间
pca = PCA(n_components=2, random_state=42)
bert_2d = pca.fit_transform(embeddings)

# Step 5: 可视化结果
plt.figure(figsize=(8, 6))
# 节点的颜色由 LDA 概率最大的主题决定
scatter = plt.scatter(bert_2d[:, 0], bert_2d[:, 1], c=np.argmax(lda_topics, axis=1), cmap='coolwarm', s=150)

# 为每个散点添加文档标签
for i, text in enumerate(texts):
    plt.annotate(f"Doc {i+1}", (bert_2d[i, 0] + 0.02, bert_2d[i, 1] + 0.02), fontsize=11)

plt.title("BERT Embeddings (PCA) Colored by LDA Main Topic")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 输出文本对应的 LDA 主题分布
print("各文本的 LDA 主题概率分布：\n", lda_topics)