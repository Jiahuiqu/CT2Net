import math
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.io import loadmat
import cv2 as cv
from torch.utils.data import DataLoader
import os
import random
from PIL import Image
import torchvision.transforms as transforms

class Dataset(nn.Module):

    def __init__(self, path, Q1, Q2, REFT, mode, C_hsi, C_rgb, padding=13):
        super(Dataset, self).__init__()
        self.mode = mode
        self.padding = padding
        self.padding_rgb = 3 * padding + 1
        self.C_hsi = C_hsi
        self.C_rgb = C_rgb
        self.image_T1 = loadmat(os.path.join(path, Q1))['T2']
        self.image_T2 = loadmat(os.path.join(path, Q2))['T1']
        self.image_REFT = loadmat(os.path.join(path, REFT))['gt']

        self.padding_image_T1 = cv.copyMakeBorder(self.image_T1, self.padding, self.padding, self.padding, self.padding,
                                                  cv.BORDER_REFLECT)
        self.padding_image_T2 = cv.copyMakeBorder(self.image_T2, self.padding_rgb, self.padding_rgb, self.padding_rgb, self.padding_rgb,
                                                  cv.BORDER_REFLECT)
        self.ht, self.wt = self.image_T2.shape[0], self.image_T2.shape[1]
        all_num = self.ht * self.wt
        self.whole_point = self.image_REFT.reshape(1, all_num)
        random.seed(1)

        hsi_gt_list = []
        for i in range(int(self.ht / 3)):
            for j in range(int(self.wt / 3)):
                # x是打了padding的高光谱
                # 取第i个训练patch，取一个立方体
                if self.image_REFT[i * 3 + 1, j * 3 + 1] == 1:
                    hsi_gt_list.append(1)
                elif self.image_REFT[i * 3 + 1, j * 3 + 1] == 0:
                    hsi_gt_list.append(0)
                else:
                    hsi_gt_list.append(2)
        self.hsi_ht = i + 1
        self.hsi_wt = j + 1
        hsi_all_num = self.hsi_ht * self.hsi_wt
        self.hsi_whole_point = np.array(hsi_gt_list).reshape(1, len(hsi_gt_list))

        if self.mode == "train":
            # Changed_point = random.sample(list(np.where(self.whole_point[0] == 1)[0]),
            #                               int(len(list(np.where(self.whole_point[0] == 0)[0]))))
            Nlabeled_point = random.sample(list(np.where(self.hsi_whole_point[0] == 0)[0]),
                                           int(len(list(np.where(self.hsi_whole_point[0] == 0)[0]))*0.1))
            NChanged_point = random.sample(list(np.where(self.hsi_whole_point[0] == 2)[0]),
                                           int(len(list(np.where(self.hsi_whole_point[0] == 2)[0])) * 0.1))
            # NChanged_point_all = np.load("/run/media/xd132/E/RJY/1.first/datasets/Bay/NChanged_point_20p.npy")
            # NChanged_point_all = NChanged_point_all.tolist()
            # NChanged_point = random.sample(NChanged_point_all,
            #                                int(len(NChanged_point_all)*0.5))
            self.random_point = NChanged_point + Nlabeled_point
            # self.random_point = NChanged_point

        if self.mode == "test":
            self.random_point = list(range(hsi_all_num))
            self.rgb_random_point = list(range(all_num))

    def __len__(self):
        ans = len(self.random_point)
        return len(self.random_point)

    def __getitem__(self, index):

        if self.mode == "train":
            scale_rgb = 2 * self.padding_rgb + 1
            # HSI坐标
            Toriginal_i = int((self.random_point[index] / self.hsi_wt))
            Toriginal_j = (self.random_point[index] - Toriginal_i * self.hsi_wt)
            Tnew_i_hsi = Toriginal_i + self.padding
            Tnew_j_hsi = Toriginal_j + self.padding

            # 对应于HSI的RGB坐标
            Toriginal_i_rgb = math.ceil(Toriginal_i * 3)
            Toriginal_j_rgb = math.ceil(Toriginal_j * 3)
            Tnew_i_rgb = Toriginal_i_rgb + self.padding_rgb
            Tnew_j_rgb = Toriginal_j_rgb + self.padding_rgb

            hsi = self.padding_image_T1[Tnew_i_hsi - self.padding: Tnew_i_hsi + 1 + self.padding,
                  Tnew_j_hsi - self.padding: Tnew_j_hsi + self.padding + 1, :]

            rgb = self.padding_image_T2[Tnew_i_rgb - self.padding_rgb: Tnew_i_rgb + 1 + self.padding_rgb,
                  Tnew_j_rgb - self.padding_rgb: Tnew_j_rgb + self.padding_rgb + 1, :]

            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow((hsi[:, :, [14, 25, 36]]))
            #
            # plt.subplot(1, 2, 2)
            # plt.imshow(rgb[:, :, :])
            # plt.show()


            hsi = hsi.transpose(2, 0, 1).reshape(self.C_hsi, self.padding*2+1, self.padding*2+1)
            rgb = rgb.transpose(2, 0, 1).reshape(self.C_rgb, self.padding_rgb*2+1, self.padding_rgb*2+1)

            return hsi, rgb, self.image_REFT[Toriginal_i, Toriginal_j]

        if self.mode == "test":
            scale_rgb = 2 * self.padding_rgb + 1
            # HSI坐标
            Toriginal_i = int((self.random_point[index] / self.hsi_wt))
            Toriginal_j = (self.random_point[index] - Toriginal_i * self.hsi_wt)
            Tnew_i_hsi = Toriginal_i + self.padding
            Tnew_j_hsi = Toriginal_j + self.padding

            # 对应于HSI的RGB坐标
            Toriginal_i_rgb = math.ceil(Toriginal_i * 3)
            Toriginal_j_rgb = math.ceil(Toriginal_j * 3)
            Tnew_i_rgb = Toriginal_i_rgb + self.padding_rgb
            Tnew_j_rgb = Toriginal_j_rgb + self.padding_rgb

            hsi = self.padding_image_T1[Tnew_i_hsi - self.padding: Tnew_i_hsi + 1 + self.padding,
                  Tnew_j_hsi - self.padding: Tnew_j_hsi + self.padding + 1, :]

            rgb = self.padding_image_T2[Tnew_i_rgb - self.padding_rgb: Tnew_i_rgb + 1 + self.padding_rgb,
                  Tnew_j_rgb - self.padding_rgb: Tnew_j_rgb + self.padding_rgb + 1, :]

            hsi = hsi.transpose(2, 0, 1).reshape(self.C_hsi, self.padding*2+1, self.padding*2+1)
            rgb = rgb.transpose(2, 0, 1).reshape(self.C_rgb, self.padding_rgb*2+1, self.padding_rgb*2+1)

            return hsi, rgb, self.image_REFT[Toriginal_i, Toriginal_j], Tnew_i_hsi, Tnew_j_hsi, Tnew_i_rgb, Tnew_j_rgb


if __name__ == "__main__":
    Cin1 = 224
    Cin2 = 3
    path = '/run/media/xd132/E/RJY/1.first/datasets/Bay'
    db = Dataset(path, 'T1HSI.mat', 'T2RGB.mat', 'REF.mat', 'train', 224, 3)
    train_data = DataLoader(db, batch_size=1, shuffle=True)
    a = 0
    for step, (T1, T2, REFT) in enumerate(train_data):
        print("hsi:{} rgb:{}".format(T1.shape, T2.shape))
        # a = a+1
        # print(a)
