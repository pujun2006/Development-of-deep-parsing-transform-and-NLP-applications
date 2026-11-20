import random
from collections import defaultdict
import jieba

# 输入示例文本
text = "深度学习模型通过多层非线性变换学习数据的特征。自然语言处理包括语言生成、机器翻译、文本分类等任务。机器学习方法不断演化，以适应更复杂的应用场景。"

# 分词
tokens = jieba.lcut(text)

# 定义n-gram模型的n值
n = 3

# 构建n-gram模型
n_grams = defaultdict(list)
for i in range(len(tokens) - n):
    key = tuple(tokens[i:i + n - 1])
    next_word = tokens[i + n - 1]
    n_grams[key].append(next_word)

# 定义生成文本的函数
def generate_text(n_grams, num_words=30):
    # 随机选择n-gram的起始词
    start = random.choice(list(n_grams.keys()))
    result = list(start)
    for _ in range(num_words - len(start)):
        state = tuple(result[-(n - 1):])  # 获取最后n-1个词作为状态
        next_word_choices = n_grams.get(state, [])
        if not next_word_choices:
            break
        next_word = random.choice(next_word_choices)
        result.append(next_word)
    return ' '.join(result)

# 生成文本
generated_text = generate_text(n_grams)
print("生成的长文本:", generated_text)
