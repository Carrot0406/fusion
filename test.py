import os
import torch
import torch.utils.data
from torchvision import transforms
from build_dataset import build_result_dataset
from tools import para_name
import opts
import torch.nn as nn
import build_model
from classifier import Classifier  # 你需要导入正确的分类器类

# 设置测试选项
opt = opts.opt_algorithm()
opt.modality = 'c'
# opt.modality = 'v'
# opt.size_img = [810, 1440]
opt.size_img = [1080, 1920]

# 设定是否使用 CUDA
CUDA = 1  # 1 for True; 0 for False
# 指定使用的 GPU 设备索引
device_index = 1  # 指定使用第二张 GPU 设备

# 设置 PyTorch 使用的 GPU 设备
torch.cuda.set_device(device_index)

# 加载模型
model = build_model.build(CUDA, opt)
model_path = '/home/bailu/fusion/result/net_c_only=pointnet~method=attention_only~lr=0.0005/model_best.pt'
# model_path = '/home/bailu/fusion/result/net_v_only=resnet50~method=attention_only~bs=2~decay=4~lr=0.0005~lrd_rate=0.05/model_best.pt'

model.load_state_dict(torch.load(model_path),strict=False)  # 替换为你的模型文件路径
model.eval()
if CUDA:
    model.cuda()

# 创建测试数据集
transform_img_test = transforms.Compose([
    transforms.Resize(opt.size_img),
    transforms.ToTensor(),
])
dataset_test = build_result_dataset(opt, 'test', transform_img_test)
# 创建数据加载器
kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=False, **kwargs)

# 定义评估函数
def test_epoch():
    count = 0
    final_loss_all = 0
    with torch.no_grad():
        for batch_idx, (data_rgb, data_depth,data_cloud,pathloss,index) in enumerate(test_loader):
            criterion = nn.MSELoss()
            # criterion = nn.L1Loss()
            if CUDA:
                data_cloud = data_cloud.cuda()
                data_rgb = data_rgb.cuda()
                # data_depth2 = data_depth2.cuda()
                data_depth = data_depth.cuda()
                pathloss = pathloss.cuda()

            # data_img = torch.cat((data_rgb, data_depth,data_depth2), dim=1)
            data_img = torch.cat((data_rgb, data_depth), dim=1)
            data_img = data_img.to(torch.float)
            data_cloud = data_cloud.squeeze(1)
            data_cloud = data_cloud.to(torch.float)
            
            if opt.modality == 'v':
                out = model(data_img)
                print(out.shape)
            if opt.modality == 'c':
                out = model(data_cloud)
            if opt.modality == 'v+c':
                out = model(data_img, data_cloud)
            # print(out.shape)
            # print(f'Batch {batch_idx + 1}, Out shape: {out.shape}')
            print(pathloss.shape)
            final_loss = criterion(out, pathloss)
            log_test.write(
                f"Loss: {final_loss.item()}\n")
            log_test.flush()
            final_loss_all = final_loss_all + final_loss.item()
            count = count + 2
            print(count, final_loss_all)


            #将pl写回txt文件
            for i in range(2):
                time_index = index[i] # 拿到文件序号
                current_out = out[i]
                # print(time_index)
                import re
                numbers = re.findall('\d+',str(time_index))
                number = numbers[-1]
                path = '/home/bailu/fusion/test_result/0918/concat/pl'
                if not os.path.exists(path):
                    os.makedirs(path)
                result_filename = os.path.join(path,f'pl_{number}.txt')
                with open(result_filename, 'w') as result_filename:
                    for row in current_out.cpu().numpy():
                        # print(row)
                        result_filename.write(str(row) + '\n')

    final_loss = final_loss_all / count
    log_test.write(
                f"final loss: {final_loss_all}\n")
    log_test.write(
                f"avg loss: {final_loss}\n")
    log_test.flush()

    return final_loss_all, final_loss


if __name__ == '__main__':
    test_result = './test_result/0918/concat'
    if not os.path.exists(test_result):
        os.makedirs(test_result)
    log_test = open(os.path.join(test_result, 'log_test.csv'), 'w')
    
    final_loss_all, final_loss = test_epoch()
    print(final_loss_all,final_loss)
    print("Testing completed. Results saved in individual text files.")