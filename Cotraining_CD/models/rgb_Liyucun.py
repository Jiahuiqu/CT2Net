import torch
import torch.nn as nn
import scipy.io as sio
from CBAM import CBAMBlock

class Ex_Net(nn.Module):
    def __init__(self, channl):
        super(Ex_Net, self).__init__()
        self.conv1 = nn.Conv2d(channl, 64, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        # self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        # self.maxpool2 = nn.MaxPool2d(2, 2)
        # self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        # self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 128, 3, 2, 1)
        self.maxpool3 = nn.MaxPool2d(2, 2)


    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.maxpool1(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.relu(self.maxpool2(out))
        # out = self.relu(self.conv5(out))
        # out = self.relu(self.maxpool3(out))
        # out = self.relu(self.conv5(out))
        # out = self.relu(self.maxpool3(out))
        out = self.relu(self.conv5(out))

        return out


class BCNN_rgb_Liyucun(nn.Module):
    def __init__(self, channl=3):
        super(BCNN_rgb_Liyucun, self).__init__()
        self.cnn1 = Ex_Net(channl)
        self.cnn2 = Ex_Net(channl)
        self.linear3 = nn.Linear(128, 2)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(0.8)
        self.feat1 = CBAMBlock(128)  # 1111111111111111111
        self.feat2 = CBAMBlock(128)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.epsilon = 1e-7

    def forward(self, T1, T2):
        T1 = self.cnn1(T1*10)
        T2 = self.cnn2(T2*10)
        T1 = self.feat1(T1)  # 1111111111111111111
        T2 = self.feat1(T2)
        diff_feature = T1.transpose(2, 3).matmul(T2)
        # out = self.relu(self.conv5(diff_feature))
        out = diff_feature
        out = self.maxpool3(out)
        out = out.view(T1.size(0), -1)
        out1 = torch.where(out < 0, torch.ones_like(out) * -1, torch.ones_like(out))
        out = torch.sqrt(torch.abs(out))
        out = out1 * out
        out = torch.div(out, torch.norm((out+self.epsilon), 2, 1, True))
        # out = self.linear3(self.linear2(self.linear1(out)))
        out = self.linear3(out)
        return out, diff_feature

if __name__ == "__main__":
    x = torch.rand(1,3,49,49)
    y = torch.rand(1,3,49,49)
    model = BCNN_rgb(3)
    out, diff_feature = model(x,y)






