word2vecAndBert.py代码是一个综合性的实战演练脚本，它将前面介绍的 **Word2Vec（静态词嵌入）** 与 **BERT（动态上下文嵌入）** 进行了全方位的横向对比与应用演示。代码涵盖了 **Word2Vec 模型训练、相似度计算**、**BERT 句子向量提取**、**单字/词的嵌入形态对比** 以及 **基于两种模型的句子级聚合与距离度量**。

下面分板块对这段代码进行详细拆解：

---

### 1. 数据准备与 Word2Vec 训练模块

```python
# 定义示例文本
sentences = [
    ["自然", "语言", "处理", "是", "人工智能", "的", "重要", "组成"],
    ["机器", "学习", "是", "人工智能", "的", "核心", "领域"],
    ["深度", "学习", "广泛", "应用", "于", "图像", "处理", "和", "语音", "识别"]
]

# 使用Skip-gram训练Word2Vec模型，设定词向量维度为50
word2vec_model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1, epochs=100)

```

* **扩充语料**：相比前一个例子，这里提供了 3 个更为丰富的已分词句子。
* **训练配置**：
* `vector_size=50`：将每个词映射到 50 维空间。
* `epochs=100`：迭代训练 100 轮，使静态词向量的分布更趋于稳定。



---

### 2. Word2Vec 特性应用：相似度与最近邻检索

```python
# 使用 Word2Vec 模型计算相似度
similarity = word2vec_model.wv.similarity("自然", "语言")
print("词语 '自然' 与 '语言' 的相似度:", similarity)

# 使用 Word2Vec 查找与 '人工智能' 相似的词
similar_words = word2vec_model.wv.most_similar("人工智能", topn=3)
print("与 '人工智能' 最相似的词:", similar_words)

```

* **`similarity`**：基于词向量的余弦相似度（Cosine Similarity），计算两个词在向量空间中的夹角余弦值，评估其语义关联性。
* **`most_similar`**：在词表中检索与“人工智能”向量距离最近的 Top-N 个词汇。

---

### 3. BERT 嵌入提取模块

```python
# 初始化 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

```

* 定义了标准 BERT 特征提取函数，输入任意文本，经过分词、Transformer 前向传播后，截取首个位置的 `[CLS]` 标记对应的隐层向量作为**整句的语义表征**。

---

### 4. 核心对比分析：Word2Vec 与 BERT 词嵌入的区别

```python
def compare_embeddings(word2vec_model, bert_model, tokenizer, words):
    print("词嵌入对比分析结果：")
    for word in words:
        # Word2Vec 嵌入
        word2vec_embedding = word2vec_model.wv[word]

        # BERT 嵌入
        inputs = tokenizer(word, return_tensors="pt")
        outputs = bert_model(**inputs)
        bert_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
        ...

```

* **核心对比点**：
* **Word2Vec 嵌入**：从训练好的查找表（Lookup Table）中直接按索引取出固定向量（维度为 50）。
* **BERT 嵌入**：将单独一个词作为输入送入深层 Transformer 网络。**注意：** 此时 BERT 提取出的词嵌入是**动态受上下文影响**的（即使只输入一个孤立的词，其内部的多头注意力机制和位置编码也会参与计算，且其维度是 `bert-base` 的标准 768 维）。



---

### 5. 综合应用：句向量聚合与距离度量

由于 Word2Vec 是**词级别**的，无法直接得到句向量，代码通过“求平均”的方式将句子中的词向量进行聚合：

```python
# 定义一个函数，计算文本的平均 Word2Vec 嵌入
def average_word2vec_embedding(text, model):
    embedding = sum([model.wv[word] for word in text if word in model.wv]) / len(text)
    return embedding

```

* **平均池化（Mean Pooling）**：将一句话中所有词的 Word2Vec 向量相加求平均，作为整句话的粗略静态表征。

随后，通过欧氏距离公式评估句际差异：

```python
def euclidean_distance(embedding1, embedding2):
    return torch.dist(embedding1, embedding2, p=2).item()

bert_emb1 = get_bert_embedding("自然语言处理是人工智能的重要组成")
bert_emb2 = get_bert_embedding("深度学习在图像识别中的应用")
distance = euclidean_distance(bert_emb1, bert_emb2)

```

* 利用 PyTorch 的 `torch.dist` 计算两个 BERT 句向量之间的欧氏距离（Euclidean Distance），数值越小说明语义越接近。

---

### 6. 最终的大规模文本对照库

最后一部分代码构建了一个包含 4 个句子的语料库：

```python
text_corpus = [
    "自然语言处理是人工智能的重要组成",
    "机器学习是人工智能的核心",
    "深度学习在图像识别中的应用",
    "数据科学与机器学习"
]

```

它对每一个句子分别用两种方式提取向量：

1. **BERT 嵌入**（基于 Transformer 深度上下文注意力机制生成的 768 维整句 `[CLS]` 向量）。
2. **平均 Word2Vec 嵌入**（基于静态词向量简单的线性平均生成的 50 维句向量）。

这段代码直观地展现了从浅层静态词向量（Word2Vec）**向**深层动态上下文向量（BERT）演进的技术路线，以及它们在处理句子和文本相似度计算时的不同实现方式。
