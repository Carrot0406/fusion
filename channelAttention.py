import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()

        # 全局平均池化
        y = self.avg_pool(x).view(b, c)

        # 通道注意力机制
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)

        # 重塑注意力权重并应用于输入特征图
        y = y.view(b, c, 1, 1)
        out = x * y

        return out

# 示例使用
input_channels = 64
reduction_ratio = 16

# 创建通道注意力模块
channel_attention = ChannelAttention(input_channels, reduction_ratio)

# 随机生成输入特征图
batch_size = 32
height, width = 64, 64
input_features = torch.randn(batch_size, input_channels, height, width)

# 使用通道注意力机制处理输入特征图
output_features = channel_attention(input_features)

# 打印输出特征图的形状
print(output_features.shape)