import torch
import torch.nn as nn
import scipy.io as sio
from CBAM import CBAMBlock
import numpy as np
device = torch.device("cuda:1")

class Ex_Net(nn.Module):
    def __init__(self, channl):
        super(Ex_Net, self).__init__()
        self.conv1 = nn.Conv2d(channl, 64, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.5)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        # self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        # self.maxpool2 = nn.MaxPool2d(2, 2)


    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.maxpool1(out))
        out = self.relu(self.conv3(out))
        # out = self.relu(self.conv4(out))
        # out = self.maxpool2(out)
        return out

class colearning_layer(nn.Module):
    def __init__(self, bz=16, channel=128):
        super(colearning_layer, self).__init__()
        self.channel = channel
        self.weight = nn.Parameter(torch.full((bz, channel, channel), 0.001))
        # self.weight = nn.Parameter(torch.randn(bz, channel, channel))
        self.linear_e = nn.Linear(channel, channel, bias=False)
        self.epsilon = 1e-7

    def forward(self, df_hsi, df_rgb):
        nan_mask = torch.isnan(df_rgb)
        df_rgb[nan_mask] = self.epsilon
        fea_size = df_hsi.size()[2:]
        df_rgb_flat = df_rgb.view(-1, self.channel, fea_size[0] * fea_size[1])  # N,C,H*W
        df_rgb_t = torch.transpose(df_rgb_flat, 1, 2).contiguous()  # batch size x dim x num
        df_rgb_corr = self.linear_e(df_rgb_t)
        df_hsi_flat = df_hsi.view(-1, self.channel, fea_size[0] * fea_size[1])
        S = torch.bmm(df_rgb_corr, self.weight)
        S = torch.bmm(S, df_hsi_flat)
        Norm = torch.sum(torch.exp(S), dim=2)
        A = torch.exp(S)/(Norm.unsqueeze(-1))
        df_att = torch.bmm(df_rgb_flat, A).contiguous()

        att = df_att.view(-1, self.channel, fea_size[0], fea_size[1])
        nan_mask = torch.isnan(att)
        att[nan_mask] = 0.0
        return att

class BCNN_hsi_Liyucun(nn.Module):
    def __init__(self, channl=194):
        super(BCNN_hsi_Liyucun, self).__init__()
        self.cnn1 = Ex_Net(channl)
        self.cnn2 = Ex_Net(channl)
        self.linear = nn.Linear(128, 2)
        self.relu = nn.LeakyReLU(0.01)
        # self.relu = nn.ReLU(inplace=True)
        self.feat1 = CBAMBlock(128)  # 1111111111111111111
        self.feat2 = CBAMBlock(128)

        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.epsilon = 1e-7

    def forward(self, T1, T2, diff_feature_rgb):
        T1 = self.cnn1(T1)
        T2 = self.cnn2(T2)
        # T1 = self.feat1(T1)  # 1111111111111111111
        # T2 = self.feat1(T2)
        out = T1.transpose(2, 3).matmul(T2)
        diff_feature_hsi = nn.functional.interpolate(out, (3, 3))
        B, C = diff_feature_hsi.size()[:2]
        colearning = colearning_layer(bz=B, channel=C).to(device)
        att = colearning(diff_feature_hsi, diff_feature_rgb)
        out = diff_feature_hsi + att
        # out = self.relu(self.conv5(out))
        out = self.relu(self.maxpool3(out))
        out = out.view(T1.size(0), -1)
        out1 = torch.where(out < 0, torch.ones_like(out) * -1, torch.ones_like(out))
        out = torch.sqrt(torch.abs(out))
        out = out1 * out
        out = torch.div(out, torch.norm((out+self.epsilon), 2, 1, True))
        out = self.linear(out)
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(16,194,5,5).to(device)
    y = torch.rand(16,194,5,5).to(device)
    diff = torch.rand(16,128,3,3).to(device)
    model = BCNN_hsi_Liyucun(194).to(device)
    out = model(x,y,diff)
    print(out.shape)






