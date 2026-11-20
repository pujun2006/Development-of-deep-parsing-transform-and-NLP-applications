import spacy
# 加载SpaCy的中文模型（若使用英文句子，请切换至英文模型）
nlp = spacy.load("en_core_web_sm")

# 示例文本
text = "The chef prepared a delicious meal for the guests."

# 处理文本，生成依存关系树
doc = nlp(text)

# 输出依存关系中的主谓宾结构和修饰关系
print("Word\tDependency\tHead\tRelation")
for token in doc:
    print(f"{token.text}\t{token.dep_}\t{token.head.text}\t{[child.text for child in token.children]}")

# 打印语法树结构并标注主谓宾和修饰关系
print("\nDependency Tree:")
for token in doc:
    # 输出主谓宾结构
    if token.dep_ in {"nsubj", "ROOT", "dobj"}:
        print(f"{token.text} ({token.dep_}) <-- {token.head.text} ({token.head.dep_})")
    # 输出修饰关系
    elif token.dep_ in {"amod", "advmod", "prep", "pobj"}:
        print(f"{token.text} ({token.dep_}) modifies --> {token.head.text} ({token.head.dep_})")

