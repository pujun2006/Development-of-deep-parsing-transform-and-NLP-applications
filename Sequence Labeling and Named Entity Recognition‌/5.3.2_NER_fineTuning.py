import os
import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# 1. 设置国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 加载 Fast 分词器（支持 word_ids）
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# 3. 稳健载入 CoNLL-2003 数据集（适配最新 datasets 纯数据流）
try:
  dataset = load_dataset("conll2003", trust_remote_code=False)
except Exception:
  try:
    dataset = load_dataset("conll2003", name="conll2003")
  except Exception:
    data_files = {
        "train": (
            "https://hf-mirror.com/datasets/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003/train/0000.parquet"
        ),
        "validation": (
            "https://hf-mirror.com/datasets/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003/validation/0000.parquet"
        ),
        "test": (
            "https://hf-mirror.com/datasets/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003/test/0000.parquet"
        ),
    }
    dataset = load_dataset("parquet", data_files=data_files)

# 获取标签列表
label_list = dataset["train"].features["ner_tags"].feature.names
print("✅ 数据集加载成功！标签数:", len(label_list))


# 4. 数据预处理与标签对齐
def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(
      examples["tokens"], truncation=True, is_split_into_words=True
  )
  labels = []
  for i, label in enumerate(examples["ner_tags"]):
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    label_ids = [
        -100 if word_id is None else label[word_id] for word_id in word_ids
    ]
    labels.append(label_ids)
  tokenized_inputs["labels"] = labels
  return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 5. 加载 BERT 模型与训练参数
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_list)
)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",  # 👈 加上这行，彻底禁止 Trainer 自动调用 wandb 等日志工具
)

# 6. 指标评估逻辑
metric = evaluate.load("seqeval")


def compute_metrics(p):
  predictions, labels = p
  predictions = np.argmax(predictions, axis=2)

  true_labels = [
      [label_list[l] for l in label if l != -100] for label in labels
  ]
  true_predictions = [
      [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]

  results = metric.compute(
      predictions=true_predictions, references=true_labels
  )
  return {
      "precision": results["overall_precision"],
      "recall": results["overall_recall"],
      "f1": results["overall_f1"],
      "accuracy": results["overall_accuracy"],
  }


# 7. 动态 Padding 与 Trainer 初始化
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. 启动训练
trainer.train()
eval_results = trainer.evaluate()
print("\nEvaluation results:", eval_results)