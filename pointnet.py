# pointnet++代码
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torch


class get_point_model(nn.Module):
    def __init__(self, normal_channel=False):
        super(get_point_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(2048, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(1024, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, 4096)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(4096, 5159)
        self.relu6 = nn.ReLU()

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu6(x)

        return x

if __name__ == '__main__':
    model = get_point_model()
    point_cloud = torch.randn(64,3,8000)
    # import ipdb
    # ipdb.set_trace()
    feature = model(point_cloud)
    print(feature)
