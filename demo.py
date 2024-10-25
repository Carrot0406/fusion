import torch

# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
    print("CUDA 可用")
    
    # 打印 CUDA 设备数量
    num_devices = torch.cuda.device_count()
    print(f"发现 {num_devices} 个 CUDA 设备:")
    
    # 打印每个 CUDA 设备的名称和能力
    for i in range(num_devices):
        device = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print(f"设备 {i}: {device}, 计算能力: {capability}")
else:
    print("CUDA 不可用")

import torch

import torch

# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
    print("发现 CUDA 设备，尝试使用...")
    # 创建一个随机张量
    x = torch.rand(3, 3)
    print("创建张量成功，准备将其移到 CUDA 设备...")
    
    # 将张量移到 CUDA 设备上
    x = x.cuda()
    print("张量移动到 CUDA 设备成功，正在执行计算...")
    
    # 在 CUDA 设备上执行一些操作
    y = torch.matmul(x, x.t())
    print("计算完成，准备将结果移到 CPU 并打印...")
    
    # 将结果移到 CPU 上并打印
    print(y.cpu())
    print("结果打印完成。")
else:
    print("CUDA 不可用")

