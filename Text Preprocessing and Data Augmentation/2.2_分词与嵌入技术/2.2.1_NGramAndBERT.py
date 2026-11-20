import re
from nltk import ngrams
from transformers import BertTokenizer

# 示例文本
text = "自然语言处理是人工智能的重要领域。"

# 定义n-gram分词函数
def generate_ngrams(text, n):
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)

    # 切分为单词列表
    #words = text.split()
    words = list(text)

    # 生成n-grams
    n_grams = list(ngrams(words, n))
    # 将n-gram转为字符串形式
    ngram_strings = [' '.join(gram) for gram in n_grams]

    return ngram_strings

# 二元和三元分词
bigrams = generate_ngrams(text, 2)
trigrams = generate_ngrams(text, 3)

# BERT分词
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_tokens = bert_tokenizer.tokenize(text)

# 输出结果
print("原始文本:", text)
print("二元分词结果:", bigrams)
print("三元分词结果:", trigrams)
print("BERT分词结果:", bert_tokens)
