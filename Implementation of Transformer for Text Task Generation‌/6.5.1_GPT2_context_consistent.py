import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 开启国内镜像源加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前运行设备: {device}")

# 2. 加载专为对话微调的中文轻量模型 (仅 0.5B，速度极快)
model_name = "Qwen/Qwen1.5-0.5B-Chat"

print("正在加载对话模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32  # CPU 模式下使用 float32，有 GPU 可改为 float16
).to(device)
print("模型加载完成！\n")

# 3. 标准的对话历史结构 (List of Dicts)
messages = [
    {"role": "user", "content": "你好，请介绍一下自己。"},
    {"role": "assistant", "content": "你好，我是 AI 助手，可以回答你的各种问题。"},
]


# 4. 对话生成函数
def chat(user_input):
    # 将用户最新发言追加到对话历史中
    messages.append({"role": "user", "content": user_input})

    # 使用大模型标准的 Chat Template 拼接 Prompt (会自动加上系统角色标记)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 编码为 Tensor
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # 模型生成回复
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,  # 生成的最大新 Token 数
            do_sample=True,  # 开启采样
            top_p=0.8,
            temperature=0.7,  # 控制创造力
            repetition_penalty=1.1,  # 机制上防止“嘞嘞嘞”等重复
        )

    # 精确截取回答部分的 Token
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(
            model_inputs.input_ids, generated_ids
        )
    ]

    # 解码为文本
    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # 将助手的正确回答追加到历史中，以便进行下一轮对话
    messages.append({"role": "assistant", "content": response})

    return response


# ==================== 测试运行 ====================

# 第一轮对话测试
query1 = "那你都能做些什么？"
print(f"用户: {query1}")
response1 = chat(query1)
print(f"助手: {response1}\n")

# 第二轮对话测试 (模型会记住之前的上文)
query2 = "能帮我写一首关于秋天的诗吗？"
print(f"用户: {query2}")
response2 = chat(query2)
print(f"助手: {response2}")