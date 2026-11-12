import torch
from sentence_transformers import SentenceTransformer, util

# 核心修复：更换为支持中文的多语言模型（或 'shibing624/text2vec-base-chinese'）
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 定义一组句子用于计算相似度
sentences = [
    "机器学习是人工智能的一个重要分支。",
    "人工智能包含机器学习和深度学习技术。",
    "今天的天气很好，适合外出游玩。",
    "下周的天气可能会下雨。"
]

# 将句子转化为嵌入向量
embeddings = model.encode(sentences, convert_to_tensor=True)

# 计算余弦相似度矩阵
cosine_similarities = util.pytorch_cos_sim(embeddings, embeddings)

# 打印余弦相似度矩阵
print("句子间余弦相似度矩阵：")
print(cosine_similarities)


# 定义函数来获取最相似的句子
def find_most_similar(sentence_idx, sentences, cosine_similarities, top_n=3):
    """
    输出与指定句子最相似的句子及其相似度。
    """
    similarities = cosine_similarities[sentence_idx]
    top_results = torch.topk(similarities, k=top_n + 1)  # 包含自身

    print(f"\n与句子 '{sentences[sentence_idx]}' 最相似的句子:")
    for idx, score in zip(top_results.indices[1:], top_results.values[1:]):  # 跳过自身
        print(f"句子: {sentences[idx]}, 相似度: {score:.4f}")


# 展示每个句子最相似的句子
for i in range(len(sentences)):
    find_most_similar(i, sentences, cosine_similarities)
