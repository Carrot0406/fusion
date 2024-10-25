# 为了保持验证集的各个文件顺序一致
# 输入文件路径
input_file = '/home/bailu/fusion/dataset/28GHz/split/test/sub6_Lidar_test.txt'
# 输出文件路径
output_file = '/home/bailu/fusion/dataset/28GHz/split/test/sub6_pl_test.txt'

# 打开输入文件以读取每一行内容
with open(input_file, 'r') as f:
    lines = f.readlines()

# 创建一个空列表，用于存储替换后的内容
modified_lines = []

# 遍历每一行内容，并进行替换
for line in lines:
    # 使用字符串的 replace() 方法进行替换
    modified_line = line.replace('/home/bailu/dataset/Lidar/sunny_sub6_mediumVTD/RSF5_train', '/home/bailu/fusion/dataset/sub6').replace('_pointcloud', '_pl')
    # 将替换后的行添加到列表中
    modified_lines.append(modified_line)

# 打开输出文件以写入内容
with open(output_file, 'w') as f:
    # 将替换后的内容写入文件
    f.writelines(modified_lines)
