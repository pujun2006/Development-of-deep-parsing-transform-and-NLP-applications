import pandas as pd
import numpy as np
import json

# 示例标签数据，包含多个标签
data = {
    "id": [1, 2, 3],
    "text": [
        "自然语言处理是一门重要的学科",
        "深度学习在图像识别领域表现优异",
        "机器学习可以用于预测股票价格"
    ],
    "tags": [
        ["NLP", "人工智能", "计算机科学"],
        ["深度学习", "计算机视觉", "人工智能"],
        ["机器学习", "金融", "预测"]
    ]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 获取所有唯一标签并生成标签索引字典
all_tags = sorted(set(tag for sublist in df['tags'] for tag in sublist))
tag_to_index = {tag: idx for idx, tag in enumerate(all_tags)}

# Multi-Hot编码
def multi_hot_encode(tags, tag_to_index):
    multi_hot_vector = np.zeros(len(tag_to_index), dtype=int)
    for tag in tags:
        if tag in tag_to_index:
            multi_hot_vector[tag_to_index[tag]] = 1
    return multi_hot_vector

# 为每个样本生成Multi-Hot编码
df['multi_hot'] = df['tags'].apply(lambda tags: multi_hot_encode(tags, tag_to_index))

# 输出标签编码字典和每个样本的编码结果
print("标签编码字典:", tag_to_index)
print("\n每个样本的Multi-Hot编码:")
print(df[['id', 'text', 'multi_hot']])

# 存储优化：将编码后的标签矩阵保存为二进制格式
multi_hot_matrix = np.stack(df['multi_hot'].values)
np.save('multi_hot_labels.npy', multi_hot_matrix)

# 读取并展示存储的二进制标签矩阵
loaded_multi_hot_matrix = np.load('multi_hot_labels.npy')
print("\n加载后的Multi-Hot编码矩阵:")
print(loaded_multi_hot_matrix)

# 将DataFrame存储为CSV
df.drop(columns=['tags', 'multi_hot']).to_csv('data_with_labels.csv', index=False, encoding='utf-8')
print("\n已将数据和标签编码保存至CSV文件: data_with_labels.csv")