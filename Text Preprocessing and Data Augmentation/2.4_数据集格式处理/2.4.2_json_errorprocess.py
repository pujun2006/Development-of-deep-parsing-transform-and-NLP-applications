import json
import pandas as pd


def process_json_array(file_path, required_fields):
    valid_data = []
    error_log = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)  # 一次性加载整个JSON数组

        for line_num, record in enumerate(records, start=1):
            # 确保元素是字典类型（防止JSON里混入非对象数据）
            if isinstance(record, dict):
                # 检查是否包含必要字段
                if all(field in record for field in required_fields):
                    valid_data.append(record)
                else:
                    error_log.append({"line": line_num, "error": "Missing required fields", "data": record})
            else:
                error_log.append({"line": line_num, "error": "Invalid Record Format", "data": record})

    except json.JSONDecodeError as e:
        print(f"整个JSON文件解析失败，请检查文件语法错误: {e}")

    return valid_data, error_log


# 设置所需字段
required_fields = ["id", "text", "tags"]

# 处理JSON数组
valid_data, error_log = process_json_array('data_large.json', required_fields)

# 将有效数据转换为DataFrame便于后续处理
df = pd.DataFrame(valid_data)

print("有效数据：")
print(df)

print("\n错误日志：")
for error in error_log:
    print(f"索引/行号 {error['line']} - 错误类型: {error['error']} - 数据: {error['data']}")
