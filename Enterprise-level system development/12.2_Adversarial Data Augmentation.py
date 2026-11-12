import random
from opencc import OpenCC
from nlpcda import Similarword

# 替代 synonyms 的近义词替换组件
smw = Similarword(create_num=1, change_rate=0.3)

data = [
    {"text": "这款产品非常好用，功能强大且易操作", "label": "positive"},
    {"text": "服务态度差，体验非常糟糕", "label": "negative"},
    {"text": "产品质量一般，但价格实惠", "label": "neutral"}
]

# 近义词替换
def synonym_replacement(text):
    res = smw.replace(text)
    return res[0] if res else text

# 简繁体转换
cc = OpenCC('s2t')
def convert_simplified_to_traditional(text):
    return cc.convert(text)

# 随机同音/近音字错别字
def typo_augmentation(text, typo_prob=0.2):
    typo_dict = {'好': '号', '差': '查', '强': '墙', '易': '依'}
    words = list(text)
    for i, word in enumerate(words):
        if random.random() < typo_prob and word in typo_dict:
            words[i] = typo_dict[word]
    return ''.join(words)

# 扩充数据
def augment_data(data):
    augmented_data = []
    for entry in data:
        text = entry['text']
        label = entry['label']

        augmented_data.append({"text": text, "label": label})
        augmented_data.append({"text": synonym_replacement(text), "label": label})
        augmented_data.append({"text": convert_simplified_to_traditional(text), "label": label})
        augmented_data.append({"text": typo_augmentation(text), "label": label})

    return augmented_data

augmented_data = augment_data(data)
for entry in augmented_data:
    print(f"Text: {entry['text']}, Label: {entry['label']}")