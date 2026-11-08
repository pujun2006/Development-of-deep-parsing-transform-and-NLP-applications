import torch
from transformers import BertTokenizer, BertModel
from gensim.models import Word2Vec

# 示例文本数据
sentences = [
    ["自然", "语言", "处理", "是", "人工智能", "的", "重要", "组成"],
    ["机器", "学习", "是", "人工智能", "的", "核心", "领域"]
]

# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1)  # 使用Skip-gram模型

# 获取Word2Vec词嵌入
word2vec_nlp = word2vec_model.wv['自然']
word2vec_ai = word2vec_model.wv['人工智能']

# 初始化BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

# 定义BERT词嵌入提取函数
def get_bert_embedding(text):
    # 分词并转为ID
    inputs = tokenizer(text, return_tensors="pt")
    # 获取BERT嵌入
    outputs = bert_model(**inputs)
    # 获取[CLS]标记的嵌入表示
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# 获取BERT嵌入
text = "自然语言处理是人工智能的重要组成"
bert_embedding = get_bert_embedding(text)

# 输出结果
print("Word2Vec '自然' 词嵌入:", word2vec_nlp)
print("Word2Vec '人工智能' 词嵌入:", word2vec_ai)
print("BERT句子嵌入:", bert_embedding)
