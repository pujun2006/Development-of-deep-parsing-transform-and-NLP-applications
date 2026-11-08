import json
import pandas as pd


# 示例：逐行读取JSON文件并处理错误数据
def process_large_json(file_path, required_fields):
    valid_data = []
    error_log = []

    with open(file_path, 'r', encoding='utf-8') as f:
        line_num = 1
        for line in f:
            try:
                # 尝试加载当前行的JSON数据
                record = json.loads(line)

                # 检查是否包含必要字段
                if all(field in record for field in required_fields):
                    valid_data.append(record)
                else:
                    # 记录缺失字段的数据
                    error_log.append({"line": line_num, "error": "Missing required fields", "data": line})
            except json.JSONDecodeError:
                # 记录JSON格式错误
                error_log.append({"line": line_num, "error": "JSON Decode Error", "data": line})
            line_num += 1

    # 输出校验通过的数据和错误日志
    return valid_data, error_log


# 设置所需字段
required_fields = ["id", "text", "tags"]

# 处理大JSON文件并获取有效数据和错误日志
valid_data, error_log = process_large_json('data_large.json', required_fields)

# 将有效数据转换为DataFrame便于后续处理
df = pd.DataFrame(valid_data)

print("有效数据：")
print(df.head())

print("\n错误日志：")
for error in error_log[:5]:  # 只显示前5条错误日志示例
    print(f"行号 {error['line']} - 错误类型: {error['error']} - 数据: {error['data']}")
