import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 开启国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载支持 System Prompt 的轻量中文对话模型
model_name = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32
).to(device)

# 3. 初始化上下文 (包含初始对话)
messages = [
    {"role": "user", "content": "你好，可以介绍一下你是什么吗？"},
    {
        "role": "assistant",
        "content": "你好，我是一个人工智能助手，可以回答你的各种问题。",
    },
]


# 4. 定义具备“主题/角色控制”的生成函数
def generate_response(user_input, topic_system_prompt):
    # (1) 动态更新系统指令 (System Prompt)，从而精准控制回答的主题与风格
    # 先清除上一次可能存在的 system 消息，保持上下文干净
    current_messages = [
        msg for msg in messages if msg["role"] != "system"
    ]

    # 将新的主题指令放在最顶层
    current_messages.insert(
        0, {"role": "system", "content": topic_system_prompt}
    )

    # (2) 追加用户最新提问
    current_messages.append({"role": "user", "content": user_input})

    # (3) 使用官方标准的 apply_chat_template 渲染 Prompt
    prompt = tokenizer.apply_chat_template(
        current_messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # (4) 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.85,
            temperature=0.7,
            repetition_penalty=1.1,
        )

    # (5) 安全精准截取回答部分的 Token (彻底消除切片 Bug)
    new_tokens = outputs[0][inputs.input_ids.shape[-1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # (6) 将本轮对话更新到全局历史中
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response})

    return response


# ==================== 测试运行 ====================

# 1. 编程主题对话
system_programming = "你是一位经验丰富的资深软件工程师，请用专业、简洁的方式解答编程问题。"

print("--- 场景 1：编程主题 ---")
q1 = "你可以告诉我如何学习编程吗？"
print(f"用户: {q1}")
print(f"助手: {generate_response(q1, system_programming)}\n")

q2 = "那如何提高编程水平？"
print(f"用户: {q2}")
print(f"助手: {generate_response(q2, system_programming)}\n")


# 2. 切换到健康主题对话 (通过修改 System Prompt 实现主题无缝切换)
system_health = "你是一位专业的健康管理顾问，请从作息、饮食、运动等方面给出科学的健康建议。"

print("--- 场景 2：切换到健康主题 ---")
q3 = "顺便问一下，有关保持健康的建议吗？"
print(f"用户: {q3}")
print(f"助手: {generate_response(q3, system_health)}")