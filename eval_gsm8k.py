import json
import os
import re
# import fnmatch

base_path = 'outputs/default/20240324_101010'
# n_list = [
#     "47.19B", "102.24B", "149.42B", "196.61B", "243.79B", "298.84B", "346.03B",
#     "401.08B", "448.27B", "503.32B", "550.5B", "597.69B", "652.74B", "699.92B",
#     "747.11B", "802.16B", "849.35B", "904.4B", "951.58B", "1101.0B", "1203.24B",
#     "1305.48B", "1399.85B", "1502.09B", "1604.32B", "1698.69B", "1800.93B",
#     "1903.17B", "1997.54B", "2099.77B", "2202.01B", "2304.25B", "2398.62B",
#     "2500.85B", "2595.23B", "2697.46B", "2799.7B", "2901.93B", "2996.31B"
# ]
    
# list path
n_list = []
for file in os.listdir('outputs/default/20240324_101010/predictions/exp2.6'):
    if 10.0 < float(file.replace('B','')) and float(file.replace('B','')) < 2300:
        n_list.append(file)
n_list = sorted(n_list, key=lambda x:float(x.replace('B','')))

def extract_numbers_and_compare(json_data):
    equal_count = 0

    for key, value in json_data.items():
        # Extract all numbers (including decimals) from prediction
        prediction_numbers = re.findall(r"\b\d+\.?\d*\b", value['prediction'])

        # Extract the number following "####" in gold
        gold_number_match = re.search(r"#### (\d+\.?\d*)", value['gold'])
        if gold_number_match:
            gold_number = gold_number_match.group(1)

            # Check if the gold number is in the prediction numbers and count if equal
            if gold_number in prediction_numbers:
                equal_count += 1

    return equal_count

def find_folders_containing_subdir(root_dir, subdir_string):
    matched_folders = set()  # 使用集合避免重复
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            # 构建当前遍历到的目录的完整路径
            full_path = os.path.join(root, dir_name)
            # 检查该路径是否包含特定的子目录字符串
            if subdir_string in full_path:
                matched_folders.add(full_path)  # 添加包含子目录的父目录
    return list(matched_folders)

correct_ratio_dict={}

for n in n_list:
    print(f"Processing {n}...")
    matched_folders = find_folders_containing_subdir(base_path, f'/predictions/exp2.6/{n}')

    for folder_path in matched_folders:
        print(folder_path)
        # 列出所有包含"gsm8k"的文件
        files = [f for f in os.listdir(folder_path) if 'gsm8k' in f and f.endswith('.json')]

        # 初始化统计变量
        total_correct = 0
        total_count = 0

        # 遍历文件并处理
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            
            # 读取JSON文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # 使用之前定义的函数处理数据
            correct_count = extract_numbers_and_compare(data)
            
            # 更新统计
            total_correct += correct_count
            total_count += len(data)

        if total_count == 1319:
            correct_ratio = total_correct / total_count if total_count > 0 else 0

            print(n, total_correct, total_count, correct_ratio)
            correct_ratio_dict[n] = correct_ratio

print(correct_ratio_dict)
for k, v in correct_ratio_dict.items():
    print(f"{k}: {round(v*100, 2)}")

