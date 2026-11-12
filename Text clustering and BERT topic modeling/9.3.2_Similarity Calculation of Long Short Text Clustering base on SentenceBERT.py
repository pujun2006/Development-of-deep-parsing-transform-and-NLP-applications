from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 加载Sentence-BERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 定义短文本和长文本
short_texts = [
    "Climate change impacts weather patterns",
    "Global warming affects sea levels",
    "Artificial intelligence is evolving",
    "Machine learning is a subset of AI"
]

long_texts = [
    "Climate change is causing significant alterations in global weather patterns, leading to more frequent extreme events.",
    "The rise in global temperatures has led to the melting of polar ice caps, subsequently causing sea levels to rise and impacting coastal regions.",
    "Artificial intelligence, particularly in the field of natural language processing, is transforming how humans interact with technology.",
    "Machine learning, a subset of artificial intelligence, involves algorithms that improve through experience, driving advancements across industries."
]

# 生成嵌入
short_embeddings = model.encode(short_texts)
long_embeddings = model.encode(long_texts)

# 短文本相似度矩阵
short_similarity_matrix = cosine_similarity(short_embeddings)
print("Short Texts Cosine Similarity Matrix:\n", short_similarity_matrix)

# 长文本相似度矩阵
long_similarity_matrix = cosine_similarity(long_embeddings)
print("Long Texts Cosine Similarity Matrix:\n", long_similarity_matrix)

# 短文本和长文本之间的相似度
cross_similarity_matrix = cosine_similarity(short_embeddings, long_embeddings)
print("Cross Similarity Matrix between Short and Long Texts:\n", cross_similarity_matrix)