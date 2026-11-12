import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime import InferenceSession
import numpy as np

# Step 1: 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

device = torch.device("cpu")
model.to(device)
model.eval()

# Step 2: 定义 ONNX 导出的路径
onnx_path = "bert_model.onnx"

# Step 3: 创建用于 ONNX 导出的示例输入
dummy_input = tokenizer(
    "This is a sample input for ONNX conversion.",
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True
)

input_ids = dummy_input["input_ids"].to(device)
attention_mask = dummy_input["attention_mask"].to(device)

# Step 4: 替换为 PyTorch 原生 API 导出模型
torch.onnx.export(
    model,
    (input_ids, attention_mask),               # 传入位置参数元组
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],                    # 显式指定输出名称
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    },
    opset_version=14                           # 推荐推荐升至 14 或以上，提高算子兼容性
)

print(f"ONNX模型已导出至 {onnx_path}")

# Step 5: 使用 ONNX Runtime 加载模型并进行推理验证
session = InferenceSession(onnx_path)

onnx_inputs = {
    "input_ids": input_ids.cpu().numpy(),
    "attention_mask": attention_mask.cpu().numpy()
}

outputs = session.run(None, onnx_inputs)

logits = outputs[0]
predicted_class = np.argmax(logits, axis=1)

print("ONNX推理结果:", logits)
print("预测类别:", predicted_class)

# Step 6: 比较 PyTorch 与 ONNX 的推理结果
with torch.no_grad():
    torch_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    torch_logits = torch_outputs.logits.cpu().numpy()

print("PyTorch推理结果:", torch_logits)

# 精度断言对比
np.testing.assert_allclose(torch_logits, logits, rtol=1e-3, atol=1e-5)
print("✅ PyTorch 与 ONNX 推理结果一致！")