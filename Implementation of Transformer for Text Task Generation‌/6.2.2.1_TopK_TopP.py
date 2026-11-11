import torch
import torch.nn.functional as F

# 模拟的词汇表及其对应的概率分布
vocab = ['the', 'cat', 'sat', 'on', 'a', 'mat', 'dog', 'barked']
logits = torch.tensor([1.5, 2.3, 1.2, 0.8, 1.4, 0.6, 1.8, 1.0])  # 模拟的模型输出概率分布
probabilities = F.softmax(logits, dim=-1)  # 转换为概率分布


# Top-K采样实现
def top_k_sampling(probabilities, k):
    values, indices = torch.topk(probabilities, k)
    chosen_index = torch.multinomial(values, 1)  # 从Top-K词中随机选择一个
    return indices[chosen_index.item()]


# Top-P采样实现
def top_p_sampling(probabilities, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 获取最小的词集合，使其累积概率大于p
    idx = torch.where(cumulative_probs >= p)[0][0]

    # 从这些词中随机选择
    chosen_index = torch.multinomial(sorted_probs[:idx + 1], 1)
    return sorted_indices[chosen_index.item()]


# 执行Top-K采样（选择前3个词）
k = 3
top_k_word = top_k_sampling(probabilities, k)
print(f"Top-K采样结果: {vocab[top_k_word]}")

# 执行Top-P采样（设置累积概率阈值为0.9）
top_p_word = top_p_sampling(probabilities, 0.9)
print(f"Top-P采样结果: {vocab[top_p_word]}")

