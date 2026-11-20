import torch
import torch.nn.functional as F

# 模拟的中文词汇表及其对应的概率分布
vocab = ['我', '是', '学', '生', '在', '学', '习', '编', '程', '中', '信', '息', '技', '术', '方', '法']
logits = torch.tensor([1.2, 2.3, 1.5, 1.0, 2.5, 1.1, 2.0, 0.8, 1.9, 2.1, 0.9, 1.3, 1.7, 2.2, 1.4, 2.6])  # 模拟的模型输出概率分布
probabilities = F.softmax(logits, dim=-1)  # 转换为概率分布


# Top-K采样实现
def top_k_sampling(probabilities, k):
    values, indices = torch.topk(probabilities, k)  # 获取Top-K概率和索引
    chosen_index = torch.multinomial(values, 1)  # 从Top-K词中随机选择一个
    return indices[chosen_index.item()]


# Top-P采样实现
def top_p_sampling(probabilities, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)  # 按概率降序排序
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # 计算累积概率

    # 获取最小的词集合，使其累积概率大于p
    idx = torch.where(cumulative_probs >= p)[0][0]

    # 从这些词中随机选择
    chosen_index = torch.multinomial(sorted_probs[:idx + 1], 1)
    return sorted_indices[chosen_index.item()]


# 文本生成函数
def generate_text(probabilities, method='top_k', k=3, p=0.9, num_words=5):
    text = ''
    for _ in range(num_words):
        if method == 'top_k':
            word_idx = top_k_sampling(probabilities, k)
        elif method == 'top_p':
            word_idx = top_p_sampling(probabilities, p)

        text += vocab[word_idx.item()]  # 生成下一个字符
    return text


# 执行Top-K采样（选择前3个词）
generated_text_k = generate_text(probabilities, method='top_k', k=3, num_words=5)
print(f"Top-K采样生成文本: {generated_text_k}")

# 执行Top-P采样（设置累积概率阈值为0.9）
generated_text_p = generate_text(probabilities, method='top_p', p=0.9, num_words=5)
print(f"Top-P采样生成文本: {generated_text_p}")
