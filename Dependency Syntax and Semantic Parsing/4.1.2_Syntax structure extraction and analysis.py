import spacy
from spacy import displacy

# 加载SpaCy的语言模型
nlp = spacy.load("en_core_web_sm")

# 示例句子
text = "The quick brown fox jumps over the lazy dog."

# 处理文本，生成依存关系树
doc = nlp(text)

# 构建依存关系树并输出结果
print("Token\tDependency\tHead\tChildren")
for token in doc:
    print(f"{token.text}\t{token.dep_}\t{token.head.text}\t{[child.text for child in token.children]}")

# 使用displacy可视化依存关系树
print("\nRendering dependency tree visualization...")
displacy.render(doc, style="dep", jupyter=False, options={'distance': 90})

# 句法结构的提取示例：找出所有的主谓宾结构和修饰关系
print("\nIdentifying syntactic roles:")
for token in doc:
    # 提取主语、谓语、宾语
    if token.dep_ in {"nsubj", "ROOT", "dobj"}:
        print(f"{token.text} ({token.dep_}) <- {token.head.text} ({token.head.dep_})")
    # 提取修饰语
    elif token.dep_ in {"amod", "advmod", "prep", "pobj"}:
        print(f"{token.text} ({token.dep_}) modifies -> {token.head.text} ({token.head.dep_})")

# 进一步提取复杂结构：所有谓词的宾语及其修饰语
print("\nExtracting objects and their modifiers:")
for token in doc:
    if token.dep_ == "dobj":
        print(f"Object: {token.text}")
        for child in token.children:
            if child.dep_ in {"amod", "det"}:
                print(f"  Modifier: {child.text}")
