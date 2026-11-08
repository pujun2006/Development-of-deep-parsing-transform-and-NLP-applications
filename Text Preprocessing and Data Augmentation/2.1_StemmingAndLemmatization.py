import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# 下载WordNet词典资源
nltk.download('wordnet')
nltk.download('omw-1.4')

# 初始化词干提取器和词形还原器
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# 示例文本数据
text = ["running", "happier", "boxes", "studies", "easily", "flying"]


# 定义词干提取与词形还原函数
def stem_and_lemmatize(words):
    stemmed_words = []
    lemmatized_words = []
    for word in words:
        # 词干提取
        stemmed_word = stemmer.stem(word)
        stemmed_words.append(stemmed_word)

        # 词形还原，指定词性为动词以提高准确性
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet.VERB)
        lemmatized_words.append(lemmatized_word)

    return stemmed_words, lemmatized_words


# 清洗后的词干和词形还原结果
stemmed, lemmatized = stem_and_lemmatize(text)

# 输出结果
print("原始文本:", text)
print("词干提取结果:", stemmed)
print("词形还原结果:", lemmatized)
