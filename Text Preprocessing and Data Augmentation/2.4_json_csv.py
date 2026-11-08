import json
import pandas as pd

# ------------------------ JSON文件读取与处理 ------------------------

# JSON文件内容模拟
json_data = [
    {
        "id": 1,
        "text": "自然语言处理是一门重要的学科",
        "tags": ["NLP", "人工智能", "计算机科学"]
    },
    {
        "id": 2,
        "text": "深度学习在图像识别领域表现优异",
        "tags": ["深度学习", "计算机视觉", "人工智能"]
    },
    {
        "id": 3,
        "text": "机器学习可以用于预测股票价格",
        "tags": ["机器学习", "金融", "预测"]
    }
]

# 将JSON数据写入文件
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 提取特定字段
    texts = [entry['text'] for entry in data]
    tags = [entry['tags'] for entry in data]
    return texts, tags

# 读取JSON文件并展示
texts, tags = read_json_file('data.json')
print("JSON文件中的文本内容:", texts)
print("JSON文件中的标签:", tags)

# ------------------------ CSV文件读取与处理 ------------------------

# 创建模拟CSV数据
csv_data = {
    "id": [1, 2, 3],
    "text": ["自然语言处理是一门重要的学科", "深度学习在图像识别领域表现优异", "机器学习可以用于预测股票价格"],
    "tags": ["NLP|人工智能|计算机科学", "深度学习|计算机视觉|人工智能", "机器学习|金融|预测"]
}

# 将数据写入CSV文件
csv_df = pd.DataFrame(csv_data)
csv_df.to_csv('data.csv', index=False, encoding='utf-8')

# 读取CSV文件
def read_csv_file(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    # 分割标签字段，处理为列表形式
    data['tags'] = data['tags'].apply(lambda x: x.split('|'))
    return data

# 读取CSV文件并展示
csv_data = read_csv_file('data.csv')
print("\nCSV文件内容:")
print(csv_data)

# ------------------------ 优化读取大文件的速度 ------------------------

# 分块读取CSV文件
def read_large_csv_in_chunks(file_path, chunk_size=2):
    chunk_data = pd.DataFrame()
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8'):
        chunk['tags'] = chunk['tags'].apply(lambda x: x.split('|'))
        chunk_data = pd.concat([chunk_data, chunk], ignore_index=True)
    return chunk_data

# 模拟大文件读取
large_csv_data = read_large_csv_in_chunks('data.csv', chunk_size=1)
print("\n分块读取CSV文件内容:")
print(large_csv_data)

# ------------------------ 转换与保存JSON和CSV格式 ------------------------

# 将读取的数据保存为新的JSON文件
def save_to_json(data, file_path):
    data_dict = data.to_dict(orient='records')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

save_to_json(csv_data, 'converted_data.json')
print("\n已将CSV数据转换并保存为JSON格式文件: converted_data.json")

# 将JSON数据保存为新的CSV文件
def save_to_csv(data, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, encoding='utf-8')

save_to_csv(json_data, 'converted_data.csv')
print("\n已将JSON数据转换并保存为CSV格式文件: converted_data.csv")
