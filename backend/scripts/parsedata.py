import json
from itertools import islice

# 假设原始文件名为 'data.json'
input_filename = 'frontend/src/assets/data_dict_0212.json'
output_filename = 'frontend/src/assets/data_dict_0212_part.json'

# 读取 JSON 数据
with open(input_filename, 'r') as file:
    data = json.load(file)

first_four_items = dict(islice(data.items(), 4))

# 从每个元素中移除 'embedding' 键
for key in first_four_items.keys():
    if 'embedding' in first_four_items[key]:
        del first_four_items[key]['embedding']

# 将更新后的数据写入新的 JSON 文件
with open(output_filename, 'w') as file:
    json.dump(first_four_items, file, indent=4)

print(f"New JSON file '{output_filename}' has been created with the updated content.")
