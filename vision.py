import numpy as np
import matplotlib.pyplot as plt

# 读取文本文件中的数据
data = np.loadtxt('/home/bailu/fusion/test_result/pl_v01/pl_1.txt')

# 将数据重塑为67x77的矩阵
matrix = data.reshape(77, 67)

# 创建图像
plt.figure(figsize=(10, 8))
plt.imshow(matrix, cmap='gray', aspect='auto')

# 保存图像为文件
plt.savefig('image1.png')

# 关闭图像
plt.close()
