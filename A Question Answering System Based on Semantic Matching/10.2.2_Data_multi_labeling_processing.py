import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

# 创建示例数据
data = {
    "text1": [
        "苹果是一种广泛种植的水果",
        "深度学习是人工智能的一个分支",
        "地球是太阳系的行星之一",
        "自然语言处理研究语言和计算的交叉点",
        "鲸鱼生活在海洋中"
    ],
    "text2": [
        "苹果可以用于制作果汁",
        "机器学习和深度学习是AI的重要组成",
        "火星也是太阳系的一个行星",
        "计算语言学研究语言和算法",
        "鲨鱼是海洋中的一种掠食者"
    ],
    "label": [2, 2, 1, 1, 0]  # 2=高度相似, 1=部分相似, 0=不相似
}

df = pd.DataFrame(data)

# 拆分数据集
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# 加载支持中文的多语言模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# 核心修正：计算逐对（Paired）的对应相似度
def compute_paired_similarity(df):
    embeddings_text1 = model.encode(df["text1"].tolist(), convert_to_tensor=True)
    embeddings_text2 = model.encode(df["text2"].tolist(), convert_to_tensor=True)

    # 方式 A：只提取全量相似度矩阵的主对角线 (逐对对应值)
    # cosine_scores = torch.diag(util.cos_sim(embeddings_text1, embeddings_text2))

    # 方式 B（高效）：使用点积计算逐行余弦相似度
    cosine_scores = util.pairwise_cos_sim(embeddings_text1, embeddings_text2)
    return cosine_scores.cpu().numpy()


# 将预测分数整理回 DataFrame 进行直观展示
train_df = train_df.copy()
test_df = test_df.copy()

train_df["pred_similarity"] = compute_paired_similarity(train_df)
test_df["pred_similarity"] = compute_paired_similarity(test_df)

print("--- 训练集评估结果 ---")
print(train_df[["text1", "text2", "label", "pred_similarity"]])

print("\n--- 测试集评估结果 ---")
print(test_df[["text1", "text2", "label", "pred_similarity"]])