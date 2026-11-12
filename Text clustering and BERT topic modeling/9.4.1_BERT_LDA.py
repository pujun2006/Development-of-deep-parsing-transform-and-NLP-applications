import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 加载 BERT 模型与分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
model.eval()

texts = [
    "人工智能的发展在近年来取得了突破性进展，特别是在自然语言处理领域。",
    "气候变化和全球变暖是当今世界的重大环境问题。",
    "机器学习和深度学习已经广泛应用于各个行业。",
    "教育领域对在线学习和远程教学的需求不断增加。",
    "健康和医疗是人们关注的重要领域，特别是在疫情期间。"
]

# 定义生成 BERT 嵌入的函数
def get_bert_embeddings(texts):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()  # 使用 [CLS] 向量
            embeddings.append(cls_embedding.squeeze())
    return np.array(embeddings)

# 获取嵌入表示
embeddings = get_bert_embeddings(texts)

# 核心修复：将 BERT 向量缩放到非负区间 [0, 1]
scaler = MinMaxScaler()
non_negative_embeddings = scaler.fit_transform(embeddings)

# 定义并训练 LDA 模型
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(non_negative_embeddings)

# 输出每个主题的主要特征前 5 维展示
print("LDA 训练成功！")
print("LDA 主题成分形状:", lda.components_.shape)
for idx, topic in enumerate(lda.components_):
    print(f"主题 {idx+1} 前 5 个维度权重: {topic[:5]}")

