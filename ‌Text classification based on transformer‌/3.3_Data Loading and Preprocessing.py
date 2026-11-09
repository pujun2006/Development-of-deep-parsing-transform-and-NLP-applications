import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset

# 1. 配置国内镜像源（必须放在 import datasets 之后或最顶端，解决网络与 HfUriError 报错）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 定义自定义数据集类
class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 注意：此处移除了 re.sub(r"[^\w\s]", "", text)
        # BertTokenizer 内部会自动做 lower_case 和标点符号处理，保留标点对 SST-2 情感分类至关重要

        # 将文本转换为 BERT 输入格式
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


# 2. 修改数据集加载名称为完整命名空间 "nyu-mll/glue"
dataset = load_dataset("nyu-mll/glue", "sst2")
train_texts = dataset["train"]["sentence"]
train_labels = dataset["train"]["label"]

# 初始化 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 创建自定义数据集
train_dataset = CustomTextDataset(train_texts, train_labels, tokenizer)

# 定义 DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 检查 DataLoader 中的一个批次
for batch in train_loader:
    print("Input IDs 形状:", batch["input_ids"].shape)
    print("Attention Mask 形状:", batch["attention_mask"].shape)
    print("Labels 形状:", batch["label"].shape)
    print("\n第一个样本的 Input IDs:\n", batch["input_ids"][0])
    print("\nLabels 数据:\n", batch["label"])
    break