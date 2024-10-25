import pandas as pd
import matplotlib.pyplot as plt

# 打开文件并逐行读取内容
with open('/home/bailu/fusion/result/~net_v=resnet50~method=concat~bs=1~decay=6~lr=0.0001~lrd_rate=0.1/log_test.csv', 'r') as file:
    lines = file.readlines()

# 提取每行中以空格分隔的最后一个值作为损失值
loss_values = []
count = 0

for i, line in enumerate(lines):
    if not line.startswith('Epoch'):
        count += 1
        # print(line)
        # values = line.strip().split(',')
        # loss = float(values[-1]) / 5195
        loss = float(line)/5195*5
        loss_values.append(loss)
# 生成 loss 图像
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')

# 保存图片
plt.savefig('loss_test.png')

