# 将1500个time处理成每5帧取一个，每辆车300个time 存储到txt文件里面
import os
import re
import numpy as np


def get_number_from_filename(filename):
    numbers = re.findall(r'\d+', filename)
    # print(numbers[-1])
    return int(numbers[0]) if numbers else float('inf')  # 如果文件名中没有数字，返回无穷大

def get_files(folder_path):
    '''
    拿到一个文件夹下面合适的路径
    :param folder_path:
    :return:
    '''
    # 匹配文件名中的数字，并保存模 5 为 1 的文件路径
    file_paths = []
    file_names = os.listdir(folder_path)
    sorted(file_names,key=get_number_from_filename)
    for file_name in file_names:
        numbers = re.findall(r'\d+', file_name)  # 提取文件名中的所有数字
        if numbers:
            number = int(numbers[0])  # 提取第一个数字
            if number % 5 == 1:
                file_paths.append(os.path.join(folder_path, file_name))
    # # 对文件路径进行排序
    # sorted(file_paths, key=get_number_from_filename)
    return file_paths


base_root = r'/home/bailu/fusion/dataset/28GHz'
out_root = r'/home/bailu/fusion/dataset/28GHz/split'

# model_list = ['pathloss']
model_list = ['pathloss','depth', 'lidar','RGB']
# model_list = ['RGB']

train_list = ['Car7', 'Car9', 'Car10', 'Bus3', 'RSF8']
test_list = ['RSF5']
val_list = ['Car5']

# for model in model_list:
#     train_txt = []
#     test_txt = []
#     for train in train_list:
#         train_path = os.path.join(base_root, model, train)
#         # if model == 'lidar':
#         #     train_path = os.path.join(base_root, model, train, 'down')
#         # print(train_path)
#         single_list = get_files(train_path)
#         train_txt += single_list
#     # 将文件路径保存到 txt 文件中
#     output_file_path = model + '_train' + '.txt'
#     with open(output_file_path, 'w') as f:
#         for file_path in train_txt:
#             f.write(file_path + '\n')
for model in model_list:
    output_file_path = model + '_train' + '.txt'
    with open(output_file_path, 'a') as f:
        for train in train_list:
            train_txt = {}
            train_path = os.path.join(base_root, model, train)
            single_list = get_files(train_path)
            for file in single_list:
                name = os.path.basename(file)
                number = re.findall(r'\d+',name)
                if number:
                    number = number[0]  # 获取第一个匹配的数字
                    train_txt[int(number)] = file  # 将数字转换为整数作为字典的键
            train_sorted_keys = sorted(train_txt.keys())
            train_sorted_items = [train_txt[key] for key in train_sorted_keys]
        # print(train_sorted_items)
        # import ipdb; 
        # ipdb.set_trace()
    # 将文件路径保存到 txt 文件中
            for file_path in train_sorted_items:
                f.write(file_path + '\n')

    output_file_path = model + '_test' + '.txt'
    with open(output_file_path, 'a') as f:
        for test in test_list:
            test_txt = {}
            test_path = os.path.join(base_root, model, test)
            single_list = get_files(test_path)
            for file in single_list:
                name = os.path.basename(file)
                number = re.findall(r'\d+',name)
                if number:
                    number = number[0]  # 获取第一个匹配的数字
                    test_txt[int(number)] = file  # 将数字转换为整数作为字典的键
            test_sorted_keys = sorted(test_txt.keys())
            test_sorted_items = [test_txt[key] for key in test_sorted_keys]
    # 将文件路径保存到 txt 文件中
        
            for file_path in test_sorted_items:
                f.write(file_path + '\n')
    
    output_file_path = model + '_val' + '.txt'
    with open(output_file_path, 'a') as f:
        for val in val_list:
            val_txt = {}
            val_path = os.path.join(base_root, model, val)
            single_list = get_files(val_path)
            for file in single_list:
                name = os.path.basename(file)
                number = re.findall(r'\d+',name)
                if number:
                    number = number[0]  # 获取第一个匹配的数字
                    val_txt[int(number)] = file  # 将数字转换为整数作为字典的键
            val_sorted_keys = sorted(val_txt.keys())
            val_sorted_items = [val_txt[key] for key in val_sorted_keys]
        # 将文件路径保存到 txt 文件中
        
            for file_path in val_sorted_items:
                f.write(file_path + '\n')
