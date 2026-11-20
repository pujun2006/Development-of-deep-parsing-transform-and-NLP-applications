import re
from typing import List, Dict

class RuleBasedClassifier:
    def __init__(self):
        self.rules = []

    def add_rule(self, label: str, pattern: str, sub_rules=None):
        compiled_pattern = re.compile(pattern)
        self.rules.append({"label": label, "pattern": compiled_pattern, "sub_rules": sub_rules or []})

    def classify(self, text: str) -> List[str]:
        labels = []
        for rule in self.rules:
            if rule["pattern"].search(text):
                labels.append(rule["label"])
                for sub_rule in rule["sub_rules"]:
                    if sub_rule["pattern"].search(text):
                        labels.append(sub_rule["label"])
        return labels

# 定义关键词规则
classifier = RuleBasedClassifier()
classifier.add_rule("科技", r"(技术|人工智能|大数据|机器学习)")
classifier.add_rule("财经", r"(股票|市场|投资|金融)", [
    {"label": "股市", "pattern": re.compile(r"(股票|证券|股市)")},
    {"label": "外汇", "pattern": re.compile(r"(汇率|外汇|美元)")}
])
classifier.add_rule("体育", r"(足球|篮球|网球|比赛|运动会)", [
    {"label": "足球", "pattern": re.compile(r"(足球|射门|进球)")},
    {"label": "篮球", "pattern": re.compile(r"(篮球|扣篮|三分球)")}
])

# 测试文本数据
texts = [
    "今天的科技进步让人工智能和大数据迅速发展",
    "最新的股票市场分析表明，投资机会很大",
    "昨天的足球比赛非常激烈，进球数很多",
    "汇率波动影响了外汇投资者的决策",
    "篮球比赛中扣篮得分引发观众热烈欢呼"
]

# 执行分类
for text in texts:
    labels = classifier.classify(text)
    print(f"文本: '{text}' -> 分类: {labels}")

