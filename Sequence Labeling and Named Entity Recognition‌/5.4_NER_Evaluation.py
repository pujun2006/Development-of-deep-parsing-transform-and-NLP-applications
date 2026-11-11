import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,classification_report

# 模拟NER标签
true_labels = [
    ["O", "B-PER", "I-PER", "O", "B-LOC", "O"],
    ["O", "B-ORG", "I-ORG", "O", "B-LOC", "O"],
    ["O", "O", "B-PER", "I-PER", "O", "O"]
]

pred_labels = [
    ["O", "B-PER", "I-PER", "O", "B-LOC", "O"],
    ["O", "B-ORG", "O", "O", "B-LOC", "O"],
    ["O", "O", "B-PER", "O", "O", "O"]
]

# 将标签展平为一维数组，便于计算
true_flat = [label for seq in true_labels for label in seq]
pred_flat = [label for seq in pred_labels for label in seq]


# 计算每个类别的精确度、召回率、F1得分和准确率
precision, recall, f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average='macro')
accuracy = accuracy_score(true_flat, pred_flat)

print("准确率:", accuracy)
print("精确度:", precision)
print("召回率:", recall)
print("F1得分:", f1)

# 使用classification_report计算每个实体类别的评估指标
report = classification_report(true_flat, pred_flat, labels=["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC"], zero_division=0)

print("各类实体的性能评估:")
print(report)