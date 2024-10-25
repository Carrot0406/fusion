import torch
import torch.utils.data
import torch.nn as nn
from classifier import *
from resnet import resnet50
from pointnet import get_point_model
from torch import optim


class Fusionet(nn.Module):
    def __init__(self, img_encoder, cloud_encoder, opt):
        super(Fusionet, self).__init__()
        self.img_encoder = img_encoder
        self.cloud_encoder = cloud_encoder
        self._initialize_weights()
        self.relu = nn.LeakyReLU()
        self.method = opt.method
        # 通道数目
        # self.downsample = nn.Conv2d(35, 14, kernel_size=(1, 1))
        self.downsample = nn.Conv2d(4, 4, kernel_size=(1, 1))
        self.classifier = Classifier()
        if self.method == 'attention_only':
            self.model_attention = MultiheadAttentionConcatenation(2048, 1024, 3072, 8)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, img, cloud):
        img = self.downsample(img)
        img_latent = self.img_encoder(img)
        cloud_latent = self.cloud_encoder(cloud)
        feature = []


        if self.method == 'concat':
            feature = torch.cat((img_latent, cloud_latent), dim=1)
            print("****************")
            print(img_latent.shape)
            print(cloud_latent.shape)
            feature = feature.view(2, 3, 1024)
            out = self.classifier(feature)
        elif self.method == 'resnest':
            feature = torch.cat((img_latent, cloud_latent), dim=1)
            feature = feature.view(2, 3, 1024)
            out = self.classifier(feature)
        elif self.method == 'attention_only':
            feature = self.model_attention(img_latent, cloud_latent)
            feature = feature.view(2, 3, 1024)
            out = self.classifier(feature)
        return out


class MultiheadAttentionConcatenation(nn.Module):
    def __init__(self, input_size1, input_size2, output_size, num_heads):
        super(MultiheadAttentionConcatenation, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_size1 + input_size2, num_heads=num_heads)

        self.fc_concat = nn.Linear((input_size1 + input_size2) * 2, output_size)

    def forward(self, x1, x2):
        concatenated_feature = torch.cat([x1, x2], dim=-1)

        attn_output, _ = self.multihead_attn(concatenated_feature, concatenated_feature, concatenated_feature)

        combined_output = torch.cat([attn_output, concatenated_feature], dim=-1)

        output_feature = self.fc_concat(combined_output)

        return output_feature


class Image_only(nn.Module):
    def __init__(self):
        super(Image_only,self).__init__()
        self.resnet = resnet50(image_size=[1080, 1920], pretrained=True)
        self.downsample = nn.Conv2d(4, 4, kernel_size=(1, 1))
        self.linear = nn.Linear(2048,5159)
        self.relu = nn.ReLU()
    def forward(self,img):
        img = self.downsample(img)
        x = self.resnet(img)
        # print(x.shape)
        x = self.linear(x)
        out = x.mean(dim = 0,keepdim = True)
        out = self.relu(x)
        return out


def build(CUDA, opt):
    global model
    if opt.modality == 'v':
        print("---------------------------------")
        model = Image_only()
    elif opt.modality == 'c':
        model = get_point_model()
    elif opt.modality == 'v+c':

        # resnest_model = resnest101(pretrained=False)
        # encoder_v = resnest_model
        encoder_v = resnet50(image_size=[1080, 1920], pretrained=True)
        encoder_c = get_point_model()
        model = Fusionet(encoder_v, encoder_c, opt)
    if CUDA:
        model.cuda()
    return model


def set_optimizer(model, opt):
    optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay, lr=opt.lr)
    return optimizer


def get_updateModel(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    shared_dict = {str('encoder_v.'+ k): v for k, v in pretrained_dict.items() if str('encoder_v.'+ k) in model_dict}
    if shared_dict == {}:
        shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    for k,v in pretrained_dict.items():
        print(k)
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)

    return model

def get_updateModel2(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del shared_dict['conv1.0.weight']
    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)
    return model
