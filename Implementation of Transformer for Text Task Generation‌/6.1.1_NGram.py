import random
from collections import defaultdict
import jieba

# 输入文本
text = "机器学习是一种通过统计方法让计算机拥有学习能力的技术。这种技术可以帮助计算机从数据中提取特征、分析模式，从而完成预测任务。"

# 将文本分词
tokens = jieba.lcut(text)
print(tokens)
# 定义n-gram模型的n值
n = 3

# 构建n-gram模型
n_grams = defaultdict(list)

# 生成n-grams字典
for i in range(len(tokens) - n):
    key = tuple(tokens[i:i + n - 1])  # 使用前n-1个词作为键
    next_word = tokens[i + n - 1]     # 使用第n个词作为值
    n_grams[key].append(next_word)

print(n_grams)

# 定义生成文本函数
def generate_text(n_grams, num_words=15):
    # 随机选择n-gram的起始点
    start = random.choice(list(n_grams.keys()))
    result = list(start)

    for _ in range(num_words - len(start)):
        state = tuple(result[-(n - 1):])  # 获取最后n-1个词
        next_word_choices = n_grams.get(state, [])
        if not next_word_choices:
            break
        next_word = random.choice(next_word_choices)
        result.append(next_word)
    return ' '.join(result)

# 生成并输出文本
generated_text = generate_text(n_grams)
print("生成的文本:", generated_text)