#以下代码需要GPU加速
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments

# 1. 加载 SQuAD v2 数据集
dataset = load_dataset("rajpurkar/squad_v2")

# 2. 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

max_length = 384
doc_stride = 128


# 3. 修正后的数据预处理函数
def preprocess_data(examples):
    # 处理前后的空格
    examples["question"] = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 获取滑动窗口到原始样本的映射关系
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 通过 sample_mapping 找到对应的真实样本索引
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # 如果没有答案（SQuAD v2 无答案问题），指向 [CLS]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 区分 context 和 question 的 token 范围
            sequence_ids = inputs.sequence_ids(i)

            # 找到 context 在 token 中的起始和结束位置
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # 判断答案是否完整包含在当前窗口切片中
            if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # 将字符位置映射到 token 位置
                token_start_index = context_start
                while token_start_index <= context_end and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                token_end_index = context_end
                while token_end_index >= context_start and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# 4. 数据映射预处理（移除原始无用列）
tokenized_dataset = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=dataset["train"].column_names
)

train_dataset = tokenized_dataset["train"]
validation_dataset = tokenized_dataset["validation"]

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # 使用最新的 eval_strategy
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

# 6. 定义 Trainer 并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

trainer.train()

# 7. 单句推理测试（验证集/推导）
model.eval()
question = "What is the capital of France?"
context = "France is a country in Europe. The capital of France is Paris, known for its art, culture, and cuisine."

inputs = tokenizer(question, context, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits) + 1

predict_tokens = inputs["input_ids"][0][start_index:end_index]
answer = tokenizer.decode(predict_tokens, skip_special_tokens=True)

print("Predicted Answer:", answer)