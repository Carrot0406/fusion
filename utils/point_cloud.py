# 降采样点云
import numpy as np
import os



def read_point_cloud(file_path):
    """
    从 TXT 文件中读取点云数据。
    参数：
        file_path: 字符串，TXT 文件的路径。
    返回：
        np.ndarray，包含点云数据的数组。
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_points = len(lines)
        point_cloud = np.empty((num_points, 3), dtype=float)
        for i in range(num_points):
            line = lines[i].strip().split()
            x, y, z = map(float, line)
            point_cloud[i] = [x, y, z]
    return point_cloud

# 处理原始点云数据
def read_point_cloud1(file_path):
    """
    从 TXT 文件中读取点云数据。
    参数：
        file_path: 字符串，TXT 文件的路径。
    返回：
        np.ndarray，包含点云数据的数组。
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_points = len(lines) // 3
        point_cloud = np.empty((num_points, 3), dtype=float)
        for i in range(num_points):
            x = float(lines[i * 3].strip())
            y = float(lines[i * 3 + 1].strip())
            z = float(lines[i * 3 + 2].strip())
            point_cloud[i] = [x, y, z]
    return point_cloud


def write_point_cloud(file_path, point_cloud):
    """
    将点云数据写入到 TXT 文件中。
    参数：
        file_path: 字符串，要写入的 TXT 文件路径。
        point_cloud: np.ndarray，包含点云数据的数组。
    """
    np.savetxt(file_path, point_cloud)


def downsample_point_cloud(point_cloud, voxel_size):
    """
    对点云进行降采样。
    参数：
        point_cloud: np.ndarray，包含点云数据的数组。
        voxel_size: float，Voxel 的尺寸，用于控制降采样密度。
    返回：
        np.ndarray，降采样后的点云数据。
    """
    # 计算点云数据范围
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)

    # 计算每个维度的 Voxel 数量
    voxel_counts = ((max_coords - min_coords) / voxel_size).astype(int)

    # 将点云中的点映射到 Voxel 的索引
    voxel_indices = ((point_cloud - min_coords) / voxel_size).astype(int)

    # 使用字典存储每个 Voxel 中的点云数据
    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        if key in voxel_dict:
            voxel_dict[key].append(point_cloud[i])
        else:
            voxel_dict[key] = [point_cloud[i]]

    # 计算每个 Voxel 中心的点
    downsampled_points = []
    for key, value in voxel_dict.items():
        downsampled_points.append(np.mean(value, axis=0))

    return np.array(downsampled_points)


# 读取点云数据
# input_file_path = r"E:\PythonProject\car_pl\project\dataset\28GHz\lidar\Bus3_train\time1_pointcloud.txt"
# point_cloud = read_point_cloud(input_file_path)
#
# # 设置降采样的 Voxel 尺寸
# voxel_size = 1
#
# # 降采样点云
# downsampled_point_cloud = downsample_point_cloud(point_cloud, voxel_size)
#
# # 将降采样后的点云数据写入到文件中
# output_file_path = "downsampled_point_cloud1.txt"
# write_point_cloud(output_file_path, downsampled_point_cloud)

def downsample(inpath, outpath, voxel_size):
    """
    :param inpath:
    :param outpath:
    :param voxel_size:
    :return: 降采样一个文件
    """
    point_cloud = read_point_cloud1(inpath)
    down_point_cloud = downsample_point_cloud(point_cloud, voxel_size)
    write_point_cloud(outpath, down_point_cloud)


if __name__ == '__main__':
    base_root = r'E:\PythonProject\car_pl\project\dataset\28GHz\lidar'
    lists = os.listdir(base_root)
    for car in lists:
        if car == 'Bus3' or car == 'Car10':
            continue
        car_root = os.path.join(base_root, car)
        files = os.listdir(car_root)
        for f in files:
            in_path = os.path.join(car_root, f)
            f_new = f.replace('time', '')
            out_dir_path = os.path.join(car_root, 'down')
            # 创建文件夹（如果不存在）
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)
            out_path = os.path.join(out_dir_path, f_new)
            downsample(in_path, out_path, 1)
