import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import BertTokenizer

# 1. 模拟加载金融和医学领域数据集
data = {
    "text": [
        "The company's revenue increased by 25% last quarter.",
        "A new medication for heart disease was approved by the FDA.",
        "Stock prices fluctuate due to market uncertainty.",
        "The patient shows symptoms of a rare neurological disorder.",
        "The merger will affect the overall market strategy.",
        "Clinical trials show promising results for the cancer drug."
    ],
    "label": [0, 1, 0, 1, 0, 1]  # 建议使用数值型标签：0-finance, 1-medical
}
df = pd.DataFrame(data)

# 2. 先划分训练集与测试集（防止数据泄露）
train_df, test_df = train_test_split(df, test_size=0.33, random_state=42, stratify=df["label"])

# 3. 仅对训练集进行平衡重采样（以样本少的类别为例）
finance_train = train_df[train_df.label == 0]
medical_train = train_df[train_df.label == 1]

# 假设要对 finance 进行过采样
finance_upsampled = resample(
    finance_train,
    replace=True,
    n_samples=len(medical_train),
    random_state=42
)
train_balanced = pd.concat([finance_upsampled, medical_train])

# 4. 直接使用 BERT 分词器（保留文本原始的数字和标点！）
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 注意：传入 raw text 即可，无需手动正则过滤
train_encodings = tokenizer(
    train_balanced["text"].tolist(),
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# 5. 打印编码结果验证
print("编码后的 input_ids 维度:", train_encodings["input_ids"].shape)
print("第一个样本还原为 Token:", tokenizer.convert_ids_to_tokens(train_encodings["input_ids"][0]))