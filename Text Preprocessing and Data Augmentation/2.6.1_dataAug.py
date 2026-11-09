import random
import jieba
import nltk
from nltk.corpus import wordnet as wn

# 首次使用需要下载中文 WordNet 支持
nltk.download('omw-1.4')
nltk.download('omw-2.0')
nltk.download('wordnet')

# 示例句子
sentence = "自然语言处理是人工智能和计算机科学的重要组成部分。"


# 1. 基于 Jieba + WordNet 中文库的同义词替换
def synonym_replacement_zh(sentence, n=2):
  # 使用 jieba 切分为中文词语列表
  words = list(jieba.cut(sentence))
  new_words = words[:]
  shuffled_words = [w for w in words if len(w) > 1]  # 优先替换长度大于1的词
  random.shuffle(shuffled_words)

  num_replaced = 0
  for word in shuffled_words:
    # lang='cmn' 指定查询中文词条
    synsets = wn.synsets(word, lang='cmn')
    synonyms = set()

    for syn in synsets:
      for lemma in syn.lemmas('cmn'):
        syn_name = lemma.name()
        if syn_name != word:
          synonyms.add(syn_name)

    if synonyms:
      # 随机选择一个中文同义词进行替换
      synonym = random.choice(list(synonyms))
      new_words = [synonym if w == word else w for w in new_words]
      num_replaced += 1
      if num_replaced >= n:
        break

  return ''.join(new_words)  # 中文拼接不需要中间加空格


# 2. 中文词级句子反转
def reverse_sentence_zh(sentence):
  # 使用 jieba 进行词级切分后反转
  words = list(jieba.cut(sentence))
  reversed_words = words[::-1]
  return ''.join(reversed_words)


# ------------------ 批量测试 ------------------

sentences = [
    '机器学习可以用于分类和回归任务。',
    '深度学习的应用覆盖了计算机视觉和自然语言处理。',
    '数据增强方法可以帮助提高模型的泛化能力。',
]

print('--- 单句增强测试 ---')
print('原句:', sentence)
print('中文同义词替换:', synonym_replacement_zh(sentence))
print('中文词级反转:', reverse_sentence_zh(sentence))

print('\n--- 批量增强结果 ---')
for s in sentences:
  print('原句:', s)
  print('同义词替换:', synonym_replacement_zh(s, n=1))
  print('词级反转:', reverse_sentence_zh(s))
  print('-' * 40)
