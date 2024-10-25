# 点云可视化
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_point_cloud(file_path, mode):
    """
    从 TXT 文件中读取点云数据。
    参数：
        file_path: 字符串，TXT 文件的路径。
        mode: 字符串，点云存储模式，"single" 或 "triple"。
    返回：
        np.ndarray，包含点云数据的数组。
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if mode == "single":
            point_cloud = np.array([[float(val) for val in line.strip().split()] for line in lines])
        elif mode == "triple":
            num_points = len(lines) // 3
            point_cloud = np.empty((num_points, 3), dtype=float)
            for i in range(num_points):
                x = float(lines[i*3].strip())
                y = float(lines[i*3 + 1].strip())
                z = float(lines[i*3 + 2].strip())
                point_cloud[i] = [x, y, z]
        else:
            raise ValueError("Unsupported mode. Use 'single' or 'triple'.")
    return point_cloud

def plot_2d_point_cloud(point_cloud):
    """
    绘制二维点云。
    参数：
        point_cloud: np.ndarray，包含点云数据的数组，每行三个数字表示一个点。
    """
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Point Cloud Visualization')
    plt.axis('equal')
    plt.show()

def plot_3d_point_cloud(point_cloud):
    """
    绘制三维点云。
    参数：
        point_cloud: np.ndarray，包含点云数据的数组，每行三个数字表示一个点。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Visualization')
    plt.show()

# 示例用法
file_path = r"E:\PythonProject\car_pl\project\utils\downsampled_point_cloud1.txt"

# 读取每行三个数字表示一个点的点云数据
point_cloud_triple = read_point_cloud(file_path, mode="single")

# 绘制二维点云
plot_2d_point_cloud(point_cloud_triple[:, :2])  # 仅使用X和Y坐标

# 绘制三维点云
plot_3d_point_cloud(point_cloud_triple)
