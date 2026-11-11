import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 开启国内镜像源加速模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 替换为轻量级中文对话模型 (替代无法识别中文的 DialoGPT)
model_name = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32
).to(device)

# 维护结构化的多轮对话历史
messages = []


def generate_response(user_input, max_new_tokens=100):
  # 将用户最新输入加入消息队列
  messages.append({"role": "user", "content": user_input})

  # 使用规范的 apply_chat_template 拼接 Prompt
  prompt = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  inputs = tokenizer([prompt], return_tensors="pt").to(device)

  # 生成回复 (使用 max_new_tokens 替代容易溢出的 max_length)
  with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # 控制本次回答生成的长度
        do_sample=True,
        top_p=0.85,
        temperature=0.7,
        repetition_penalty=1.1,  # 惩罚重复词
        pad_token_id=tokenizer.eos_token_id,
    )

  # 精确截取最新生成的回复 Token
  new_tokens = outputs[0][inputs.input_ids.shape[-1] :]
  response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

  # 将助手的回复追加到对话历史中，形成多轮记忆
  messages.append({"role": "assistant", "content": response})
  return response


# 示例对话轮次
inputs = [
    "你好，可以介绍一下自己吗？",
    "如何提高编程水平？",
    "有哪些健康建议？",
]

# 执行多轮对话
for user_input in inputs:
  response = generate_response(user_input)
  print(f"用户: {user_input}")
  print(f"助手: {response}\n")