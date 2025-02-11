import numpy as np
import torch.nn as nn
from scipy.io import loadmat
import cv2 as cv
from torch.utils.data import DataLoader
import os
import random


def GT_To_One_Hot(class_count, label):
    """
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    """
    temp = np.zeros(class_count, dtype=np.float32)
    temp[int(label)] = 1
    # if gt[i, j] != 0:
    #     temp[int(gt[i, j])] = 1

    return temp

class Dataset_rgb(nn.Module):

    def __init__(self,name1, name2, REF, dataset, mode, channel, padding = 25, nums = 500):
        super(Dataset_rgb, self).__init__()
        self.channel = channel
        self.mode = mode
        self.filename_T1 = name1
        self.filename_T2 = name2
        self.REF = REF
        self.nums = nums
        self.padding = padding
        self.dataset = dataset
        if dataset == 'River':
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['rgb']
            self.image_T2 = loadmat(os.path.join(self.filename_T2))['rgb']
            self.image_REF = loadmat(os.path.join(self.REF))['lakelabel_v1']/255
        elif dataset == 'China':
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['rgb']
            self.image_T2 = loadmat(os.path.join(self.filename_T2))['rgb']
            self.image_REF = loadmat(os.path.join(self.REF))['Binary']

        elif dataset == 'Bay':
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['rgb']
            self.image_T2 = loadmat(os.path.join(self.filename_T2))['rgb']
            self.image_REF = loadmat(os.path.join(self.REF))['HypeRvieW']
        elif dataset == 'Bar':
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['rgb']
            self.image_T2 = loadmat(os.path.join(self.filename_T2))['RGBt2_norm']
            self.image_REF = loadmat(os.path.join(self.REF))['HypeRvieW']
        elif dataset == 'Liyucun':
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['rgb']
            self.image_T2 = loadmat(os.path.join(self.filename_T2))['T1']
            # self.image_REF = loadmat(os.path.join(self.REF))['gt']
            self.image_REF = loadmat(os.path.join(self.REF))['a']

        elif dataset == 'USA':
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['rgb']
            self.image_T2 = loadmat(os.path.join(self.filename_T2))['rgb']
            self.image_REF = loadmat(os.path.join(self.REF))['Binary']

        self.data_pair = {}
        self.data_pair['T1'] = self.image_T1
        self.data_pair['T2'] = self.image_T2

        self.h, self.w = self.image_T1.shape[0], self.image_T1.shape[1]

        all_num = self.h * self.w
        random.seed(1)
        if self.dataset == 'Bay' or dataset == 'Bar' or dataset == 'Liyucun':
            self.image_REF[np.where(self.image_REF == 0)] = 3
            self.image_REF[np.where(self.image_REF == 2)] = 0
            self.image_REF[np.where(self.image_REF == 3)] = 2

        self.whole_point = self.image_REF.reshape(1, all_num)
        self.whole_REF = np.squeeze(self.whole_point)

        if self.mode == "train":
            if self.dataset == 'USA':
                Changed_point = random.sample(list(np.where(self.whole_point[0] == 1)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 1)[0])) / (all_num)) * (all_num * 0.20))))
                NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 0)[0])) / (all_num)) * (all_num * 0.20))))
                self.random_point = Changed_point + NChanged_point

            elif self.dataset == 'China':
                Changed_point = random.sample(list(np.where(self.whole_point[0] == 1)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 1)[0])) / (all_num)) * (all_num * 0.20))))
                NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 0)[0])) / (all_num)) * (all_num * 0.20))))
                self.random_point = Changed_point + NChanged_point

            elif self.dataset == 'Bar':
                Changed_point = random.sample(list(np.where(self.whole_point[0] == 1)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 1)[0])) / (all_num)) * (all_num * 0.20))))
                NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 0)[0])) / (all_num)) * (all_num * 0.20))))

                self.random_point = Changed_point + NChanged_point

            elif self.dataset == 'Bay':
                Changed_point = random.sample(list(np.where(self.whole_point[0] == 1)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 1)[0])) / (all_num)) * (all_num * 0.20))))
                NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 0)[0])) / (all_num)) * (all_num * 0.20))))
                # NChanged_point = np.load("/run/media/xd132/E/RJY/1.first/datasets/Bay/NChanged_point_20p.npy")
                # NChanged_point = NChanged_point.tolist()
                self.random_point = Changed_point + NChanged_point

            elif self.dataset == 'Liyucun':
                Changed_point = random.sample(list(np.where(self.whole_point[0] == 1)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 1)[0])) / (all_num)) * (all_num * 0.20))))
                NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 0)[0])) / (all_num)) * (all_num * 0.20))))
                self.random_point = Changed_point + NChanged_point

            elif self.dataset == 'River':
                Changed_point = random.sample(list(np.where(self.whole_point[0] == 1)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 1)[0])) / (all_num)) * (all_num * 0.20))))
                NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), int(
                    ((len(list(np.where(self.whole_point[0] == 0)[0])) / (all_num)) * (all_num * 0.20))))

                self.random_point = Changed_point + NChanged_point

        if self.mode == "test":
            self.random_point = list(range(all_num))
