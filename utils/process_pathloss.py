import os

base_root = r'E:\PythonProject\sunny_28G_中_pl\temp'
out_root = r'E:\PythonProject\car_pl\project\dataset\28GHz\pathloss'


def read_txt_and_get_last_column(file_path, output_file_path):
    last_column_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[3:]  # 跳过前三行
        for line in lines:
            columns = line.strip().split()  # 按空格分割行数据
            last_column_data.append(float(columns[-1]))  # 保存最后一列数据为浮点数

    with open(output_file_path, 'w') as output_file:
        for data in last_column_data:
            output_file.write(str(data) + '\n')


# 处理一个设备
if __name__ == '__main__':
    for i in range(1, 1501):
        root = os.path.join(base_root, 'time{}'.format(i))
        files = os.listdir(root)
        for file in files:
            global device
            if file.startswith('Dataset.pl.t001_32'):
                device = 'Car5'
            if file.startswith('Dataset.pl.t001_36'):
                device = 'Bus3'
            if file.startswith('Dataset.pl.t001_37'):
                device = 'RSF5'
            if file.startswith('Dataset.pl.t001_39'):
                device = 'Car9'
            if file.startswith('Dataset.pl.t001_45'):
                device = 'Car7'
            if file.startswith('Dataset.pl.t001_47'):
                device = 'Car10'
            if file.startswith('Dataset.pl.t001_49'):
                device = 'RSF8'
            file_path = os.path.join(root, file)
            out_dir = os.path.join(out_root,device)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_name = 'time{}_pl.txt'.format(i)
            print(device,"  ",out_name)
            read_txt_and_get_last_column(file_path,os.path.join(out_dir,out_name))