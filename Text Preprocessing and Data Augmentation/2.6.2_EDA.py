import random
import jieba
import nltk
from nltk.corpus import wordnet

# 下载所需的数据资源
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('omw-2.0', quiet=True)

# 原始中文句子
sentence = "机器学习可以用于自然语言处理和计算机视觉等领域。"


# 1. 中文同义词替换 (SR)
def synonym_replacement(words, n):
  new_words = words[:]
  # 传入 lang='cmn' 查询中文同义词
  random_word_list = list(
      set([
          word for word in words if len(word) > 1 and wordnet.synsets(word, lang='cmn')
      ])
  )
  random.shuffle(random_word_list)
  num_replaced = 0

  for random_word in random_word_list:
    synonyms = wordnet.synsets(random_word, lang='cmn')
    if synonyms:
      # 获取所有中文同义词
      cmn_lemmas = [
          lemma.name()
          for syn in synonyms
          for lemma in syn.lemmas('cmn')
          if lemma.name() != random_word
      ]
      if cmn_lemmas:
        synonym = random.choice(cmn_lemmas)
        new_words = [
            synonym if word == random_word else word for word in new_words
        ]
        num_replaced += 1
        if num_replaced >= n:
          break
  return new_words


# 2. 中文随机插入 (RI)
def random_insertion(words, n):
  new_words = words[:]
  for _ in range(n):
    add_word(new_words)
  return new_words


def add_word(new_words):
  synonyms = []
  counter = 0
  while len(synonyms) < 1 and counter < 10:
    if len(new_words) == 0:
      return
    random_word = new_words[random.randint(0, len(new_words) - 1)]
    for syn in wordnet.synsets(random_word, lang='cmn'):
      for lemma in syn.lemmas('cmn'):
        if lemma.name() != random_word:
          synonyms.append(lemma.name())
    counter += 1

  if synonyms:
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words))
    new_words.insert(random_idx, random_synonym)


# 3. 随机删除 (RD)
def random_deletion(words, p):
  if len(words) == 1:
    return words
  new_words = [word for word in words if random.uniform(0, 1) > p]
  # 防止全部删光的情况
  if len(new_words) == 0:
    return [random.choice(words)]
  return new_words


# 4. 随机交换 (RS)
def random_swap(words, n):
  new_words = words[:]
  if len(new_words) < 2:
    return new_words
  for _ in range(n):
    new_words = swap_word(new_words)
  return new_words


def swap_word(new_words):
  if len(new_words) >= 2:
    idx1, idx2 = random.sample(range(len(new_words)), 2)
    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
  return new_words


# 5. EDA 主函数
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
  # 关键点 1：使用 jieba 进行中文分词
  words = list(jieba.cut(sentence))
  num_words = len(words)
  augmented_sentences = []

  n_sr = max(1, int(alpha_sr * num_words))
  n_ri = max(1, int(alpha_ri * num_words))
  n_rs = max(1, int(alpha_rs * num_words))

  # 关键点 2：使用 ''.join() 拼接中文词语（不用空格）
  augmented_sentences.append(''.join(synonym_replacement(words, n_sr)))
  augmented_sentences.append(''.join(random_insertion(words, n_ri)))
  augmented_sentences.append(''.join(random_swap(words, n_rs)))
  augmented_sentences.append(''.join(random_deletion(words, p_rd)))

  return augmented_sentences


# 运行测试
augmented_sentences = eda(sentence)

print("原始句子:", sentence)
print("\n增强后的句子:")
labels = [
    "同义词替换 (SR)",
    "随机插入 (RI)",
    "随机交换 (RS)",
    "随机删除 (RD)",
]
for idx, (label, augmented_sentence) in enumerate(
    zip(labels, augmented_sentences)
):
  print(f"增强句子 {idx + 1} [{label}]: {augmented_sentence}")
