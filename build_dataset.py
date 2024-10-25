import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.point_cloud import read_point_cloud1
import torchvision.transforms as transforms
import torch


def default_loader_rgb(image_path):
    return Image.open(image_path).convert('RGB')


def default_loader_dep(image_path):
    return Image.open(image_path).convert('L')


def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_pl_data(file_path):
    """
    把路径损耗的值转换成tensor
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [float(line.strip()) for line in lines]
        tensor = torch.tensor(data)
    return tensor


# 抽样点云
def sample_point_cloud(point_cloud, target_size):
    indices = np.random.choice(point_cloud.shape[0], target_size, replace=True)
    sampled_points = point_cloud[indices]

    return sampled_points


# def get_cloud_numpy(path):
#     with open(path, 'r') as file:
#         lines = file.readlines()
#         num_points = len(lines) // 3
#         point_cloud = np.empty((num_points, 3), dtype=float)
#         for i in range(num_points):
#             x = float(lines[i * 3].strip())
#             y = float(lines[i * 3 + 1].strip())
#             z = float(lines[i * 3 + 2].strip())
#             point_cloud[i] = [x, y, z]
#     point_cloud_array_same = sample_point_cloud(point_cloud, 5000)
#     transposed_array = np.transpose(point_cloud_array_same)
#     return transposed_array

def get_cloud_numpy(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        num_points = len(lines)
        point_cloud = np.empty((num_points, 3), dtype=float)
        for i in range(num_points):
            line_parts = lines[i].strip().split()  # 分割每行字符串为单独的浮点数
            x = float(line_parts[0])  # 提取第一个浮点数
            y = float(line_parts[1])  # 提取第二个浮点数
            z = float(line_parts[2])  # 提取第三个浮点数
            point_cloud[i] = [x, y, z]
    point_cloud_array_same = sample_point_cloud(point_cloud, 5000)
    transposed_array = np.transpose(point_cloud_array_same)
    return transposed_array





# 图片数据集
class DatasetStage1(Dataset):
    def __init__(self, opt, mode, transform=None):
        self.mode = mode
        dir_rgb_names = opt.dataset_rgb
        dir_depth_names = opt.dataset_depth
        path_to_rgb = readlines(r'/home/bailu/fusion/dataset/28GHz/split/rgb_train.txt')
        path_to_depth = readlines(r'/home/bailu/fusion/dataset/28GHz/split/depth_train.txt')
        self.path_to_rgb = path_to_rgb
        self.path_to_depth = path_to_depth
        self.transform = transform
        self.loader_rgb = default_loader_rgb
        self.loader_dep = default_loader_dep

    def __len__(self):
        return len(self.path_to_rgb)

    def __getitem__(self, index):
        path_rgb = self.path_to_rgb[index]
        path_dep = self.path_to_depth[index]
        rgb = self.loader_rgb(path_rgb)
        depth = self.loader_dep(path_dep)
        if self.transform is not None:
            rgb = self.transform(rgb)
            depth = self.transform(depth)
        return (rgb, depth)


# 点云数据集
class DatasetStage2(Dataset):
    def __init__(self, opt, mode):
        # get labels
        self.mode = mode
        path_to_cloud = readlines(r'/home/bailu/fusion/dataset/28GHz/split/lidar_train.txt')
        self.path_to_cloud = path_to_cloud
        self.loader = default_loader_rgb

    def __getitem__(self, index):
        path_cloud = self.path_to_cloud[index]
        # cloud = read_point_cloud(path_cloud)  # 维度是n*3
        # cloud = np.transpose(cloud)  ## 转置之后维度变成3*n
        cloud = get_cloud_numpy(path_cloud)
        return cloud

    def __len__(self):
        return len(self.path_to_cloud)


class DatasetStage3(Dataset):
    def __init__(self, opt, mode, transform):
        self.mode = mode
        self.path_to_pl_gt = r'/home/bailu/fusion/dataset/28GHz/split/pathloss_train.txt'
        self.transform = transform
        self.loader_rgb = default_loader_rgb
        self.loader_dep = default_loader_dep

        self.depth_fpath = r'/home/bailu/fusion/dataset/28GHz/split/depth_train.txt'
        self.rgb_fpath = r'/home/bailu/fusion/dataset/28GHz/split/rgb_train.txt'
        self.lidar_fpath = r'/home/bailu/fusion/dataset/28GHz/split/lidar_train.txt'

        self.depth_filenames = readlines(self.depth_fpath)
        self.rgb_filenames = readlines(self.rgb_fpath)
        self.lidar_filenames = readlines(self.lidar_fpath)
        self.pathloss_filenames = readlines(self.path_to_pl_gt)

    def __getitem__(self, index):
        rgb = self.transform(self.loader_rgb(self.rgb_filenames[index]))
        depth = self.transform(self.loader_dep(self.depth_filenames[index]))
        # point_numpy = read_point_cloud1(self.lidar_filenames[index])
        # to_tensor = transforms.ToTensor()
        # cloud = to_tensor(point_numpy)
        cloud_numpy = get_cloud_numpy(self.lidar_filenames[index])
        to_tensor = transforms.ToTensor()
        cloud = to_tensor(cloud_numpy)
        pathloss = get_pl_data(self.pathloss_filenames[index])

        return rgb, depth, cloud, pathloss

    def __len__(self):
        return len(self.depth_filenames)


class test_dataset(Dataset):
    def __init__(self, opt, mode, transform):
        self.path_to_pl_gt = r"/home/bailu/fusion/dataset/28GHz/split/pathloss_val.txt"
        self.mode = mode
        self.transform = transform
        self.loader_rgb = default_loader_rgb
        self.loader_dep = default_loader_dep

        self.depth_fpath = r'/home/bailu/fusion/dataset/28GHz/split/depth_val.txt'
        self.rgb_fpath = r'/home/bailu/fusion/dataset/28GHz/split/rgb_val.txt'
        self.lidar_fpath = r'/home/bailu/fusion/dataset/28GHz/split/lidar_val.txt'

        self.depth_filenames = readlines(self.depth_fpath)
        self.rgb_filenames = readlines(self.rgb_fpath)
        self.lidar_filenames = readlines(self.lidar_fpath)
        
        self.pl_filenames = readlines(self.path_to_pl_gt)

    def __len__(self):
        return len(self.rgb_filenames)

    def __getitem__(self, index):
        rgb = self.transform(self.loader_rgb(self.rgb_filenames[index]))
        depth = self.transform(self.loader_dep(self.depth_filenames[index]))
        cloud_numpy = get_cloud_numpy(self.lidar_filenames[index])
        to_tensor = transforms.ToTensor()
        cloud = to_tensor(cloud_numpy)
        pathloss = get_pl_data(self.pl_filenames[index])
        return rgb,depth,cloud, pathloss


class result_dataset(Dataset):
    def __init__(self, opt, mode, transform):
        self.mode = mode
        self.transform = transform
        self.loader_rgb = default_loader_rgb
        self.loader_dep = default_loader_dep

        self.depth_fpath = r'/home/bailu/fusion/dataset/28GHz/split/test0918/depth_test.txt'
        self.rgb_fpath = r'/home/bailu/fusion/dataset/28GHz/split/test0918/rgb_test.txt'
        self.lidar_fpath = r'/home/bailu/fusion/dataset/28GHz/split/test0918/lidar_test.txt'
        self.path_to_pl_gt = r"/home/bailu/fusion/dataset/28GHz/split/test0918/pathloss_test.txt"
        self.pl_filenames = readlines(self.path_to_pl_gt)
        self.depth_filenames = readlines(self.depth_fpath)
        self.rgb_filenames = readlines(self.rgb_fpath)
        self.lidar_filenames = readlines(self.lidar_fpath)

    def __len__(self):
        return len(self.depth_filenames)

    def __getitem__(self, index):
        rgb = self.transform(self.loader_rgb(self.rgb_filenames[index]))
        depth = self.transform(self.loader_dep(self.depth_filenames[index]))
        point_numpy = get_cloud_numpy(self.lidar_filenames[index])
        to_tensor = transforms.ToTensor()
        cloud = to_tensor(point_numpy)
        pathloss = get_pl_data(self.pl_filenames[index])
        index = self.depth_filenames[index]

        return rgb, depth, cloud,pathloss,index

# 创建验证模型泛化性的其他情况下所需数据集
class other_dataset(Dataset):
    def __init__(self, opt, mode, transform,depth_path,rgb_path,lidar_path,pl_path):
        self.mode = mode
        self.transform = transform
        self.loader_rgb = default_loader_rgb
        self.loader_dep = default_loader_dep

        self.depth_fpath = depth_path
        self.rgb_fpath = rgb_path
        self.lidar_fpath = lidar_path
        self.path_to_pl_gt = pl_path
        self.pl_filenames = readlines(self.path_to_pl_gt)
        self.depth_filenames = readlines(self.depth_fpath)
        self.rgb_filenames = readlines(self.rgb_fpath)
        self.lidar_filenames = readlines(self.lidar_fpath)

    def __len__(self):
        return len(self.depth_filenames)

    def __getitem__(self, index):
        rgb = self.transform(self.loader_rgb(self.rgb_filenames[index]))
        depth = self.transform(self.loader_dep(self.depth_filenames[index]))
        point_numpy = get_cloud_numpy(self.lidar_filenames[index])
        to_tensor = transforms.ToTensor()
        pathloss = get_pl_data(self.pl_filenames[index])
        cloud = to_tensor(point_numpy)
        index = self.depth_filenames[index]

        return rgb, depth, cloud,pathloss,index
    


def build_dataset(opt, mode, transform):
    if opt.modality == 'v':  # to pretrain visual channel
        dataset = DatasetStage3(opt, mode, transform)
    elif opt.modality == 'c':  # to pretrain semantic channel
        # dataset = DatasetStage2(opt, mode)
        dataset = DatasetStage3(opt, mode, transform)
    elif opt.modality == 'v+c':  # to pretrain image channel
        dataset = DatasetStage3(opt, mode, transform)
    else:
        assert 1 < 0, 'Please fill the correct train stage!'

    return dataset

# 验证集
def build_test_dataset(opt, mode, transform):
    dataset = test_dataset(opt, mode, transform)
    return dataset

# 测试集
def build_result_dataset(opt, mode, transform):
    dataset = result_dataset(opt, mode, transform)
    return dataset
# 其他测试集
def build_other_dataset(opt, mode, transform,depth_path,rgb_path,lidar_path,pl_path):
    dataset = other_dataset(opt, mode, transform,depth_path,rgb_path,lidar_path,pl_path)
    return dataset

if __name__ == '__main__':
    t = get_pl_data(r'E:\PythonProject\car_pl\project\dataset\28GHz\pathloss\Bus3\time1_pl.txt')
    print(t.shape)
