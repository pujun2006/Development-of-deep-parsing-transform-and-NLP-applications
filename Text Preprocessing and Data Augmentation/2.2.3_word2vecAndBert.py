import torch
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import jieba  # 引入 jieba 分词库用于中文分词

# 定义示例文本（已分词，用于训练 Word2Vec）
sentences = [
    ["自然", "语言", "处理", "是", "人工智能", "的", "重要", "组成"],
    ["机器", "学习", "是", "人工智能", "的", "核心", "领域"],
    ["深度", "学习", "广泛", "应用", "于", "图像", "处理", "和", "语音", "识别"]
]

# --------------------------- Word2Vec 嵌入生成 ---------------------------

# 使用Skip-gram训练Word2Vec模型，设定词向量维度为50
word2vec_model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1, epochs=100)

# 获取Word2Vec词嵌入
word2vec_nlp = word2vec_model.wv['自然']
word2vec_ai = word2vec_model.wv['人工智能']

# 显示 Word2Vec 模型的词汇表
print("Word2Vec 词汇表:", list(word2vec_model.wv.index_to_key))

# 显示 '自然' 和 '人工智能' 的词向量
print("Word2Vec '自然' 词向量:", word2vec_nlp)
print("Word2Vec '人工智能' 词向量:", word2vec_ai)

# 使用 Word2Vec 模型计算相似度
similarity = word2vec_model.wv.similarity("自然", "语言")
print("词语 '自然' 与 '语言' 的相似度:", similarity)

# 使用 Word2Vec 查找与 '人工智能' 相似的词
similar_words = word2vec_model.wv.most_similar("人工智能", topn=3)
print("与 '人工智能' 最相似的词:", similar_words)

# --------------------------- BERT 嵌入生成 ---------------------------

# 初始化 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')


# 定义 BERT 嵌入提取函数
def get_bert_embedding(text):
    # 将文本分词并转换为ID
    inputs = tokenizer(text, return_tensors="pt")
    # 通过 BERT 模型获得嵌入
    outputs = bert_model(**inputs)
    # 获取句子[CLS]标记的嵌入表示
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding


# 定义示例文本
texts = [
    "自然语言处理是人工智能的重要组成",
    "深度学习在图像识别中的应用",
    "机器学习是人工智能的核心领域"
]

# 获取每个文本的 BERT 嵌入
bert_embeddings = [get_bert_embedding(text) for text in texts]

# 输出每个文本的嵌入结果
for idx, embedding in enumerate(bert_embeddings):
    print(f"文本 {idx + 1} 的 BERT 嵌入向量形状:", embedding.shape)


# --------------------------- 词嵌入对比分析 ---------------------------

# 分析 Word2Vec 嵌入与 BERT 嵌入的区别
def compare_embeddings(word2vec_model, bert_model, tokenizer, words):
    print("\n词嵌入对比分析结果：")
    for word in words:
        if word in word2vec_model.wv:
            # Word2Vec 嵌入
            word2vec_embedding = word2vec_model.wv[word]
        else:
            print(f"词汇 '{word}' 不在 Word2Vec 词表中")
            continue

        # BERT 嵌入
        inputs = tokenizer(word, return_tensors="pt")
        outputs = bert_model(**inputs)
        bert_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()

        # 输出 Word2Vec 嵌入与 BERT 嵌入
        print(f"\n词汇 '{word}' 的嵌入对比:")
        print("Word2Vec 嵌入:", word2vec_embedding)
        print("BERT 嵌入:", bert_embedding)


# 对比分析词汇嵌入效果
compare_words = ["自然", "学习", "人工智能"]
compare_embeddings(word2vec_model, bert_model, tokenizer, compare_words)


# --------------------------- 综合分析 ---------------------------

# 定义一个函数，计算文本的平均 Word2Vec 嵌入（使用 jieba 分词后求平均）
def average_word2vec_embedding(text, model):
    # 使用 jieba 对中文文本进行分词
    words = list(jieba.cut(text))
    # 筛选出在词表中的词，并计算其嵌入的平均值
    valid_embeddings = [model.wv[word] for word in words if word in model.wv]
    if len(valid_embeddings) == 0:
        return torch.zeros(model.vector_size)  # 如果没有匹配的词，返回全0向量

    embedding = sum(valid_embeddings) / len(valid_embeddings)
    return embedding


# 获取平均 Word2Vec 嵌入
text1 = "自然语言处理是人工智能的重要组成"
avg_embedding = average_word2vec_embedding(text1, word2vec_model)
print("\n文本的平均 Word2Vec 嵌入:", avg_embedding)


# 定义函数计算BERT嵌入的欧氏距离
def euclidean_distance(embedding1, embedding2):
    return torch.dist(embedding1, embedding2, p=2).item()


# 计算两个句子的 BERT 嵌入距离
bert_emb1 = get_bert_embedding("自然语言处理是人工智能的重要组成")
bert_emb2 = get_bert_embedding("深度学习在图像识别中的应用")
distance = euclidean_distance(bert_emb1, bert_emb2)
print("两个句子的 BERT 嵌入欧氏距离:", distance)

# --------------------------- 综合比较结果 ---------------------------

# 使用 BERT 和 Word2Vec 模型处理文本语料
text_corpus = [
    "自然语言处理是人工智能的重要组成",
    "机器学习是人工智能的核心",
    "深度学习在图像识别中的应用",
    "数据科学与机器学习"
]

# 获取每个文本的 BERT 嵌入和平均 Word2Vec 嵌入
bert_embeddings_corpus = [get_bert_embedding(text) for text in text_corpus]
# 修改处：直接传入整句字符串，由 average_word2vec_embedding 内部调用 jieba 分词
word2vec_embeddings_corpus = [average_word2vec_embedding(text, word2vec_model) for text in text_corpus]

# 展示每个文本的 BERT 和 Word2Vec 嵌入
for idx, (bert_embed, word2vec_embed) in enumerate(zip(bert_embeddings_corpus, word2vec_embeddings_corpus)):
    print(f"\n文本 {idx + 1} 的嵌入对比:")
    print("BERT 嵌入 (shape):", bert_embed.shape)
    print("平均 Word2Vec 嵌入:", word2vec_embed)
