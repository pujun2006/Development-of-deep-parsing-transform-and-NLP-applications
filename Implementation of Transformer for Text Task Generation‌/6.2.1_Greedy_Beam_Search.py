import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. 初始化 GPT-2 模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()


# 2. Greedy Search 修复版
def greedy_search(input_text, max_len=50):
  input_ids = tokenizer.encode(input_text, return_tensors="pt")
  generated = input_ids  # shape: [1, seq_len]

  with torch.no_grad():
    for _ in range(max_len):
      outputs = model(generated)
      next_token_logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]
      next_token = torch.argmax(
          next_token_logits, dim=-1, keepdim=True
      )  # shape: [1, 1]

      # 沿 seq 维度拼接 (dim=1)
      generated = torch.cat((generated, next_token), dim=1)

      # 判断标量数值是否等于 EOS Token ID
      if next_token.item() == tokenizer.eos_token_id:
        break

  return tokenizer.decode(generated[0], skip_special_tokens=True)


# 3. Beam Search 修复版
def beam_search(input_text, max_len=50, beam_width=3):
  input_ids = tokenizer.encode(input_text, return_tensors="pt")
  generated_sequences = [(input_ids, 0.0)]  # [(seq_tensor, cumulative_score)]

  with torch.no_grad():
    for _ in range(max_len):
      all_candidates = []
      for seq, score in generated_sequences:
        # 如果该序列已经以 EOS 结尾，不再继续扩展
        if seq[0, -1].item() == tokenizer.eos_token_id:
          all_candidates.append((seq, score))
          continue

        outputs = model(seq)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = F.log_softmax(next_token_logits, dim=-1)
        top_tokens = torch.topk(next_token_probs, beam_width, dim=-1)

        for i in range(beam_width):
          next_token = top_tokens.indices[0, i].unsqueeze(0).unsqueeze(0)  # [1, 1]
          new_seq = torch.cat((seq, next_token), dim=1)
          new_score = score + top_tokens.values[0, i].item()
          all_candidates.append((new_seq, new_score))

      # 按对数概率得分倒序排列，保留前 beam_width 个最优分支
      ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
      generated_sequences = ordered[:beam_width]

      # 如果当前所有候选分支都到了 EOS 结尾，提前终止整体搜索
      if all(
          seq[0, -1].item() == tokenizer.eos_token_id
          for seq, _ in generated_sequences
      ):
        break

  # 取综合得分最高的分支
  best_sequence = generated_sequences[0][0]
  return tokenizer.decode(best_sequence[0], skip_special_tokens=True)


# 4. 测试运行
input_text = "Once upon a time"
greedy_result = greedy_search(input_text)
beam_result = beam_search(input_text)

print("=== Greedy Search Result ===")
print(greedy_result)
print("\n=== Beam Search Result ===")
print(beam_result)

# 输入文本
input_text = "在城市的绿色和乡村的绿色之外，还有一块心灵的绿色，它"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Greedy Search
greedy_output = model.generate(input_ids, max_length=80, do_sample=False)
greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

# Beam Search
beam_output = model.generate(input_ids, max_length=80, num_beams=5, early_stopping=True)
beam_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)

# 打印生成的文本
print("Greedy Search Result:\n", greedy_text)
print("\nBeam Search Result:\n", beam_text)