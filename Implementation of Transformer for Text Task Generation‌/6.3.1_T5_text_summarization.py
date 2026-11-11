import os
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 1. 设置国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 加载 T5 预训练模型和 Tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 3. 修复：使用最新适配的单/多命名空间数据集路径
try:
  dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
except Exception:
  # 备用容错方案：直接加载 parquet 规范数据
  dataset = load_dataset("cnn_dailymail", name="3.0.0")


# 4. 数据预处理（添加 T5 专属的前缀 'summarize: '）
def preprocess_function(examples):
  # T5 模型需要特定的任务前缀
  inputs = ["summarize: " + article for article in examples["article"]]
  targets = [summary for summary in examples["highlights"]]

  model_inputs = tokenizer(
      inputs, max_length=512, truncation=True, padding="max_length"
  )
  labels = tokenizer(
      targets, max_length=150, truncation=True, padding="max_length"
  )

  # 将 Padding 字符的 Label 替换为 -100，防止计算 Loss
  labels_ids = labels["input_ids"]
  labels_ids = [
      [(l if l != tokenizer.pad_token_id else -100) for l in label]
      for label in labels_ids
  ]

  model_inputs["labels"] = labels_ids
  return model_inputs


# 批量转换数据集并移除原始文本列，防止 DataLoader collate 报错
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 设置数据格式为 PyTorch Tensor
tokenized_datasets.set_format("torch")

# 5. 构建 DataLoader（建议使用采样数据做示范，全量数据集较大）
train_dataset = tokenized_datasets["train"].select(range(100))  # 示例截取前100条
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 6. 设置训练设备与优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 7. 模型训练
epochs = 1
print(f"开始在 {device} 上训练...")
for epoch in range(epochs):
  model.train()
  total_loss = 0
  for step, batch in enumerate(train_dataloader):
    # 将数据移至 GPU/CPU
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    if (step + 1) % 5 == 0:
      print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}")

  print(
      f"Epoch {epoch+1}/{epochs} 训练完成, 平均 Loss:"
      f" {total_loss / len(train_dataloader):.4f}"
  )


# 8. 文本摘要生成推理函数
def generate_summary(input_text):
  model.eval()
  # 推理输入同样必须带有 'summarize: ' 前缀
  text_to_summarize = "summarize: " + input_text
  inputs = tokenizer(
      text_to_summarize,
      return_tensors="pt",
      max_length=512,
      truncation=True,
      padding="max_length",
  ).to(device)

  with torch.no_grad():
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )

  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return summary


# 9. 测试生成效果
test_text = dataset["test"][0]["article"]
print("\n=== 原始新闻文本 ===")
print(test_text[:300] + "...")
print("\n=== 训练后的 T5 生成摘要 ===")
print(generate_summary(test_text))