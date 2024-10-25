# 主要训练代码
import io
import os
import os.path
import time
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from build_dataset import build_dataset, build_test_dataset
import itertools
from tools import para_name, AverageMeter
# load environmental settings
from classifier import *
import opts

# from resnest import *
opt = opts.opt_algorithm()
import numpy as np

# import torch
# torch.backends.cudnn.enabled = False
# import ipdb
# -----------------------------------------------------------------dataset information--------------------------------------------------------------------

# opt.modality = 'v+c'
opt.size_img = [1080, 1920]

# --------------------------------------------------------------------settings----------------------------------------------------------------------------

# basic
CUDA = 1  # 1 for True; 0 for False
SEED = 1
measure_best = 0  # best measurement
epoch_best = 0
torch.manual_seed(SEED)
kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}
if CUDA:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# log and model paths
result_path = os.path.join(opt.result_path, para_name(opt))
if not os.path.exists(result_path):
    os.makedirs(result_path)

# train settings
EPOCHS = opt.lr_decay * 3
# EPOCHS = 50
# -------------------------------------------------------------dataset & dataloader-----------------------------------------------------------------------

transform_img_train = transforms.Compose([
    transforms.Resize([1080, 1920]),
    transforms.ToTensor(), ])
transform_img_test = transforms.Compose([
    transforms.Resize([1080, 1920]),
    transforms.ToTensor(), ])

# create dataset
dataset_train = build_dataset(opt, 'train', transform_img_train)
dataset_test = build_test_dataset(opt, 'test', transform_img_test)

# dataloader
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True, sampler=None, **kwargs)
# wrn/vgg -> batch_size may not 100
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=False, sampler=None, **kwargs)

# -----------------------------------------------------------------Model----------------------------------------------------------------------------------
# model define
import build_model

# model_classifier = Classifier().cuda()
model = build_model.build(CUDA, opt)
print(model)
# 初次训练前没有权重
# model = build_model.get_updateModel(model,
#                                     '/home/bailu/yxs/result_lr/datset=/home/bailu/dataset/RGB/rainy_mmWave_mediumVTD/~net_v=resnet50~method=concat~bs=1~decay=4~lr=0.001~lrd_rate=0.1/model_final.pt')

# model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])


optimizer = build_model.set_optimizer(model, opt)


# ----------------------------------------------------------------Train with cloud and image----------------------------------------------------------------------------------
def train_epoch(epoch, decay, optimizer, modality):
    model.train()
    total_time = time.time()

    for batch_idx, (data_rgb, data_depth, data_cloud, pl_gt) in enumerate(train_loader):
        import ipdb
        start_time = time.time()
        criterion = nn.MSELoss()
        # prediction and loss
        batch_size_cur = data_rgb.size(0)
        if CUDA:
            data_cloud = data_cloud.cuda()
            data_rgb = data_rgb.cuda()
            # data_depth2 = data_depth2.cuda()
            data_depth = data_depth.cuda()
            pl_gt = pl_gt.cuda()

        data_img = torch.cat((data_rgb, data_depth), dim=1)
        data_img = data_img.to(torch.float)
        data_cloud = data_cloud.squeeze(1)  # 删除的是没有用的通道
        data_cloud = data_cloud.to(torch.float)
        if opt.modality=='v+c':
            out = model(data_img, data_cloud)
        elif opt.modality=='c':
            out = model(data_cloud)
        else:
            out = model(data_img)

        final_loss = criterion(out,pl_gt)

        # optimization
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # Print loss for each batch
        print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {final_loss.item()}")
        log_train.write(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {final_loss.item()}\n")
        log_train.flush()
    # Print average loss for the entire epoch
    avg_loss = final_loss.item()  # You may want to calculate the average loss based on your needs.
    print(f"Epoch [{epoch}/{EPOCHS}] Average Loss: {avg_loss}")

    # You can also log the loss to a file if needed
    log_train.write(f"{epoch},{avg_loss}\n")
    log_train.flush()


# ----------------------------------------------------------------Test----------------------------------------------------------------------------------
def test_epoch(epoch):
    import ipdb
    model.eval()
    total_time = time.time()
    final_loss_all = 0
    count = 0
    with torch.no_grad():
        for batch_idx, (data_rgb, data_depth, data_cloud, pl_gt) in enumerate(test_loader):
            start_time = time.time()
            criterion = nn.MSELoss()
            # prediction and loss
            batch_size_cur = data_rgb.size(0)
            if CUDA:
                data_cloud = data_cloud.cuda()
                data_rgb = data_rgb.cuda()
                data_depth = data_depth.cuda()
                pl_gt = pl_gt.cuda()
            data_img = torch.cat((data_rgb, data_depth), dim=1)
            data_img = data_img.to(torch.float)
            data_cloud = data_cloud.squeeze(1)
            data_cloud = data_cloud.to(torch.float)
            if opt.modality=='v+c':
                out = model(data_img, data_cloud)
            elif opt.modality=='c':
                out = model(data_cloud)
            else:
                out = model(data_img)


            # compute loss
            final_loss = criterion(out, pl_gt)
            final_loss_all = final_loss_all + final_loss.item()
            count = count + 2

            print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(test_loader)}] Loss: {final_loss.item()}")
            log_test.write(
                f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(test_loader)}] Loss: {final_loss.item()}\n")
            log_test.flush()
    final_loss = final_loss_all / count
    log_test.write(f"{final_loss}\n")
    log_test.flush()
    return final_loss_all


def test_only():
    model.eval()
    total_time = time.time()
    final_loss_all = 0
    count = 0
    with torch.no_grad():
        for batch_idx, (data_rgb, data_depth, data_cloud, pl_gt) in enumerate(test_loader):
            start_time = time.time()
            criterion = nn.MSELoss()
            # prediction and loss
            batch_size_cur = data_rgb.size(0)
            if CUDA:
                data_cloud = data_cloud.cuda()
                data_rgb = data_rgb.cuda()
                data_depth = data_depth.cuda()
                pl_gt = pl_gt.cuda()
            data_img = torch.cat((data_rgb, data_depth), dim=1)
            data_img = data_img.to(torch.float)
            data_cloud = data_cloud.squeeze(1)
            data_cloud = data_cloud.to(torch.float)
            if opt.modality=='v+c':
                out = model(data_img, data_cloud)
            elif opt.modality=='c':
                out = model(data_cloud)
            else:
                out = model(data_img)
            # compute loss
            final_loss = criterion(out, pl_gt)
            final_loss_all = final_loss_all + final_loss.item()
            count = count + 2
            print(count, final_loss_all)

    final_loss = final_loss_all / count

    return final_loss_all, final_loss


def lr_scheduler(epoch, optimizer, lr_decay_iter, decay_rate):
    if not (epoch % lr_decay_iter):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr'] * decay_rate


if __name__ == '__main__':
    log_train = open(os.path.join(result_path, 'log_train.csv'), 'w')
    log_test = open(os.path.join(result_path, 'log_test.csv'), 'w')
    log_cls_train = open(os.path.join(result_path, 'log_train_cls.csv'), 'w')
    measure_best = 9999999999
    for epoch in range(1, EPOCHS + 1):
        lr_scheduler(epoch, optimizer, opt.lr_decay, opt.lrd_rate)
        
        train_epoch(epoch, opt.lr_decay, optimizer, opt.modality)
        str_model = '/model_1' + str(epoch) + '.pt'
        torch.save(model.state_dict(), result_path + str_model)
        measure_cur = test_epoch(epoch)
        # save current model
        if measure_cur < measure_best:
            torch.save(model.state_dict(), result_path + '/model_best.pt')
            measure_best = measure_cur
            epoch_best = epoch

        if epoch==EPOCHS:
            torch.save(model.state_dict(), result_path + '/model_final.pt')

    state = 'Max is achieved on epoch {} with Top1:{}'.format(epoch_best, measure_best)

    log_test.write(state)
    log_test.flush()

    final_loss_all,final_loss = test_only()
    print(final_loss_all,final_loss)




