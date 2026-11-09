import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from gensim.models import Word2Vec

# 1. 扩充数据集（每个类别 4 条，共 12 条）
texts = [
    # 科技类 (4条)
    "人工智能在金融市场中的应用日益广泛",
    "机器学习和数据科学的结合推动了科技创新",
    "深度学习模型在自然语言处理中效果显著",
    "大语言模型正在改变软件开发的方式",
    # 体育类 (4条)
    "足球和篮球是全球最受欢迎的运动",
    "网球比赛十分激烈场上选手表现出色",
    "篮球赛事中三分球表现出色",
    "乒乓球世界杯比赛吸引了大量观众",
    # 财经类 (4条)
    "股市波动对投资者有很大影响",
    "外汇市场的变化使得投资回报更具挑战性",
    "数据分析在金融行业中非常重要",
    "中央银行公布了最新的货币政策与利率调整"
]
labels = [
    "科技", "科技", "科技", "科技",
    "体育", "体育", "体育", "体育",
    "财经", "财经", "财经", "财经"
]

# 2. 分层划分数据集 (test_size=0.25 即测试集 3 条，刚好科技、体育、财经各 1 条)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# 打印划分情况，直观查看
print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
print("测试集真实标签分布:", y_test)
print("-" * 50)


# --- 3. TF-IDF + Linear SVM ---
def chinese_tokenizer(text):
    return list(jieba.cut(text))

tfidf_vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None)
svm_model = SVC(kernel='linear', C=1.0)
pipeline = Pipeline([('tfidf', tfidf_vectorizer), ('svm', svm_model)])

pipeline.fit(X_train, y_train)
y_pred_svm = pipeline.predict(X_test)

print("=== TF-IDF + SVM 分类结果 ===")
print("准确率:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, zero_division=0))


# --- 4. Word2Vec + 逻辑回归 ---
X_train_segmented = [list(jieba.cut(text)) for text in X_train]
X_test_segmented = [list(jieba.cut(text)) for text in X_test]

# 训练 Word2Vec 词向量
w2v_model = Word2Vec(
    sentences=X_train_segmented,
    vector_size=50,
    window=3,
    min_count=1,
    epochs=100
)

def get_word2vec_embeddings(segmented_texts, embedding_model):
    embeddings = []
    for words in segmented_texts:
        word_vectors = [embedding_model.wv[word] for word in words if word in embedding_model.wv]
        if word_vectors:
            text_embedding = np.mean(word_vectors, axis=0)
        else:
            text_embedding = np.zeros(embedding_model.vector_size)
        embeddings.append(text_embedding)
    return np.array(embeddings)

X_train_w2v = get_word2vec_embeddings(X_train_segmented, w2v_model)
X_test_w2v = get_word2vec_embeddings(X_test_segmented, w2v_model)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_w2v, y_train)
y_pred_w2v = lr_model.predict(X_test_w2v)

print("\n=== Word2Vec + 逻辑回归分类结果 ===")
print("准确率:", accuracy_score(y_test, y_pred_w2v))
print(classification_report(y_test, y_pred_w2v, zero_division=0))
