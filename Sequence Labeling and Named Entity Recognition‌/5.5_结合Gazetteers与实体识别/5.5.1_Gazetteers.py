import pandas as pd
import re

# 构建领域特定词典
# 假设构建一个包含地名和人名的词典
gazetteers = {
    "Location": ["New York", "California", "Texas", "Beijing", "Shanghai", "Paris", "Tokyo"],
    "Person": ["Alice", "Bob", "Charlie", "David", "Eve"]
}

# 将词典数据转换为DataFrame便于管理和查找
gazetteer_df = pd.DataFrame([(key, value) for key in gazetteers for value in gazetteers[key]], columns=["Entity_Type", "Entity"])

# 显示词典内容
print("构建的词典内容：")
print(gazetteer_df)

# 定义待处理的文本
text = "Alice and Bob traveled from New York to California last week, while Charlie stayed in Paris."


# 通过词典匹配找到实体
def find_entities(text, gazetteer_df):
    entities = []
    for _, row in gazetteer_df.iterrows():
        # 使用正则表达式查找词典中每个实体
        matches = re.finditer(rf"\b{re.escape(row['Entity'])}\b", text)
        for match in matches:
            entity_info = {
                "Entity": match.group(0),
                "Type": row["Entity_Type"],
                "Start": match.start(),
                "End": match.end()
            }
            entities.append(entity_info)
    return entities


# 查找并输出文本中的实体
entities_found = find_entities(text, gazetteer_df)
print("\n在文本中找到的实体：")
for entity in entities_found:
    print(f"实体: {entity['Entity']} | 类型: {entity['Type']} | 开始位置: {entity['Start']} | 结束位置: {entity['End']}")


# 示例NER预测结果 (假设)
ner_predictions = [
    {"Entity": "Alice", "Type": "Person", "Start": 0, "End": 5},
    {"Entity": "California", "Type": "Location", "Start": 38, "End": 48}
]


# 合并词典匹配结果与NER预测结果
def integrate_gazetteer_with_ner(ner_predictions, gazetteer_entities):
    combined_results = {tuple((item['Start'], item['End'])): item for item in ner_predictions}

    # 合并词典匹配结果
    for entity in gazetteer_entities:
        key = (entity["Start"], entity["End"])
        if key not in combined_results:
            combined_results[key] = entity

    # 返回合并后的结果
    return list(combined_results.values())


# 结合词典和模型结果
combined_results = integrate_gazetteer_with_ner(ner_predictions, entities_found)

# 显示最终的NER和词典匹配结果
print("\n结合词典和NER模型的实体识别结果：")
for result in combined_results:
    print(f"实体: {result['Entity']} | 类型: {result['Type']} | 开始位置: {result['Start']} | 结束位置: {result['End']}")
