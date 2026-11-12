import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# 假设数据集包含两列： 'text'（用户评论文本）和 'label'（情感标签，如 0 表示负向，1 表示中性，2 表示正向）
data = pd.DataFrame({
    "text": ["我很喜欢这个产品！", "服务态度差", "质量不错，但是有点贵", "非常满意，下次还会购买"],
    "label": [2, 0, 1, 2]
})

# 数据预处理
def preprocess_data(data, max_length=128):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")  # 使用中文BERT
    # 对数据进行编码
    encodings = tokenizer(data['text'].tolist(), truncation=True, padding=True, max_length=max_length)
    return encodings

# 加载数据并分为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_encodings = preprocess_data(train_data)
test_encodings = preprocess_data(test_data)

# 转换为Dataset格式
train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": train_data['label'].tolist()})
test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"], "labels": test_data['label'].tolist()})

# 模型初始化
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    eval_strategy="epoch",     # 每个epoch后进行评估
    per_device_train_batch_size=8,   # 每个设备的训练批次大小
    per_device_eval_batch_size=8,    # 每个设备的评估批次大小
    num_train_epochs=3,              # 训练的总epoch数
    report_to="tensorboard",            # 日志保存目录
)

# 使用 Trainer API 进行训练和评估
trainer = Trainer(
    model=model,                         # 模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset            # 测试数据集
)

# 模型训练
trainer.train()
# 模型评估
eval_results = trainer.evaluate()
print(f"模型评估结果: {eval_results}")