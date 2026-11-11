import os
import torch
from datasets import load_dataset
from torch.optim import AdamW  # 修复 1：改用 PyTorch 原生 AdamW
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 1. 设置国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 加载 T5 模型和 Tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 3. 加载数据集（修复 3：使用兼容的命名空间路径）
try:
  dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
except Exception:
  dataset = load_dataset("cnn_dailymail", name="3.0.0")


# 4. 数据预处理函数
def preprocess_function(examples):
  inputs = ["summarize: " + article for article in examples["article"]]
  targets = examples["highlights"]

  model_inputs = tokenizer(
      inputs, max_length=512, truncation=True, padding="max_length"
  )
  labels = tokenizer(
      targets, max_length=150, truncation=True, padding="max_length"
  )

  # 修复 4：将 pad_token_id 替换为 -100，屏蔽无意义 Padding 的 Loss 计算
  labels_ids = labels["input_ids"]
  labels_ids = [
      [(l if l != tokenizer.pad_token_id else -100) for l in label]
      for label in labels_ids
  ]

  model_inputs["labels"] = labels_ids
  return model_inputs


# 修复 2：移除原始文本列并设置 PyTorch 格式
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
tokenized_datasets.set_format("torch")

# 构建 DataLoader（示范时建议截取少部分数据快速验证）
train_dataset = tokenized_datasets["train"].select(range(100))
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 5. 设置设备与优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 6. 训练循环
epochs = 1
for epoch in range(epochs):
  model.train()
  for step, batch in enumerate(train_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    if (step + 1) % 5 == 0:
      print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}")


# 7. 生成摘要推理函数
def generate_summary(input_text):
  model.eval()
  input_text = "summarize: " + input_text
  inputs = tokenizer(
      input_text,
      return_tensors="pt",
      max_length=512,
      truncation=True,
      padding="max_length",
  ).to(device)

  with torch.no_grad():
    # 修复 5：显式传入 attention_mask，避免 Padding 干扰编码器
    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )

  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return summary


# 8. 测试生成
test_text = dataset["test"][0]["article"]
print("Original Article:")
print(test_text[:300] + "...")
print("\nGenerated Summary:")
print(generate_summary(test_text))