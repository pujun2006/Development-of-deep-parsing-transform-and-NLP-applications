import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForQuestionAnswering, DistilBertForQuestionAnswering, AutoTokenizer
from torch.optim import AdamW

# 1. 加载微调好的教师模型与未训练/待蒸馏的学生模型
teacher_model_name = "csarron/bert-base-uncased-squad-v1"
student_model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# 必须使用带 QA Head 的 ForQuestionAnswering 模型结构
teacher_model = BertForQuestionAnswering.from_pretrained(teacher_model_name)
student_model = DistilBertForQuestionAnswering.from_pretrained(student_model_name)

# 冻结教师模型参数
teacher_model.eval()

# 2. 定义蒸馏超参数
temperature = 2.0  # 温度系数，平滑软目标分布
alpha = 0.5  # 软硬损失平衡因子

optimizer = AdamW(student_model.parameters(), lr=3e-5)

# 模拟输入数据（问题 + 上下文）
texts = [
    ("What is Machine Learning?",
     "Machine learning is the study of computer algorithms that improve automatically through experience."),
    ("What does NLP stand for?",
     "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and AI.")
]
# 假定真实标签的起止位置 (start_pos, end_pos)
labels = [(10, 12), (0, 3)]

# 3. 标准蒸馏训练循环
student_model.train()
for epoch in range(3):
    total_loss = 0.0
    for (question, context), (start_pos, end_pos) in zip(texts, labels):
        optimizer.zero_grad()

        # 统一编码（注意：这里保持默认，让 teacher 正常接收）
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)

        # 真实标签 Tensor
        targets_start = torch.tensor([start_pos])
        targets_end = torch.tensor([end_pos])

        # A. 获取教师模型的软预测 (logits) —— BERT 正常接收 inputs
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            t_start_logits = teacher_outputs.start_logits
            t_end_logits = teacher_outputs.end_logits

        # B. 获取学生模型的预测 —— 必须剔除 DistilBERT 不支持的 token_type_ids
        student_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        student_outputs = student_model(**student_inputs)

        s_start_logits = student_outputs.start_logits
        s_end_logits = student_outputs.end_logits

        # C.1 计算硬损失 (Hard Loss / Task Loss)：学生与真实标签的交叉熵
        loss_hard_start = F.cross_entropy(s_start_logits, targets_start)
        loss_hard_end = F.cross_entropy(s_end_logits, targets_end)
        loss_hard = (loss_hard_start + loss_hard_end) / 2.0

        # C.2 计算软损失 (Soft Loss)：学生 logits 与教师 logits 的 KL 散度 (带温度 T)
        p_s_start = F.log_softmax(s_start_logits / temperature, dim=-1)
        p_t_start = F.softmax(t_start_logits / temperature, dim=-1)
        loss_soft_start = F.kl_div(p_s_start, p_t_start, reduction="batchmean") * (temperature ** 2)

        p_s_end = F.log_softmax(s_end_logits / temperature, dim=-1)
        p_t_end = F.softmax(t_end_logits / temperature, dim=-1)
        loss_soft_end = F.kl_div(p_s_end, p_t_end, reduction="batchmean") * (temperature ** 2)

        loss_soft = (loss_soft_start + loss_soft_end) / 2.0

        # D. 总蒸馏损失组合（已修正 1 - alpha 的减号遗漏）
        loss = alpha * loss_hard + (1 - alpha) * loss_soft

        # E. 反向传播与优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Average Distillation Loss: {total_loss / len(texts):.4f}")