from __future__ import absolute_import
import numpy as np
import cv2 as cv


class Preprocessor(object):
    def __init__(self, dataset, mode):
        super(Preprocessor, self).__init__()
        assert len(dataset) >= 2
        self.data_pairs = dataset[0]
        self.random_point = dataset[1]
        self.image_REF = dataset[2]
        # if type(dataset[2]) is np.ndarray:
        #     self.image_REF = np.squeeze(dataset[2].reshape(1, -1))
        self.channel = dataset[3]
        self.padding = dataset[4]

        self.image_T1 = self.data_pairs['T1']
        self.image_T2 = self.data_pairs['T2']
        self.h, self.w = self.image_T1.shape[0], self.image_T1.shape[1]

        self.padding_image_T1 = cv.copyMakeBorder(self.image_T1, self.padding, self.padding, self.padding, self.padding, cv.BORDER_REFLECT)
        self.padding_image_T2 = cv.copyMakeBorder(self.image_T2, self.padding, self.padding, self.padding, self.padding, cv.BORDER_REFLECT)

        self.mode = mode
        if mode == 'train':
            if len(dataset) > 2:
                assert len(dataset[5]) == len(dataset[2])
                self.weights = dataset[5]
            else:
                self.weights = np.ones(len(dataset[2]), dtype=np.float32)

    def __len__(self):
        return len(self.random_point)

    def __getitem__(self, index):
        original_i = int((self.random_point[index] / self.w))
        original_j = (self.random_point[index] - original_i * self.w)
        new_i = original_i + self.padding
        new_j = original_j + self.padding
        label = self.image_REF[index]

        if self.mode == 'train':
            weight = self.weights[index]
            T1 = self.padding_image_T1[new_i - self.padding: new_i + 1 + self.padding,
                 new_j - self.padding: new_j + self.padding + 1, :].transpose(2, 0, 1).reshape(self.channel,
                                                                                               self.padding * 2 + 1,
                                                                                               self.padding * 2 + 1)
            T2 = self.padding_image_T2[new_i - self.padding: new_i + 1 + self.padding,
                 new_j - self.padding: new_j + self.padding + 1, :].transpose(2, 0, 1).reshape(self.channel,
                                                                                               self.padding * 2 + 1,
                                                                                               self.padding * 2 + 1)
            label = label
            data_pair = {}
            data_pair['T1'] = T1
            data_pair['T2'] = T2
            return {'data_pair': data_pair,
                    'label': label,
                    'weight': weight}

        else:
            T1 = self.padding_image_T1[new_i - self.padding: new_i + 1 + self.padding,
                 new_j - self.padding: new_j + self.padding + 1, :].transpose(2, 0, 1).reshape(self.channel,
                                                                                               self.padding * 2 + 1,
                                                                                               self.padding * 2 + 1)
            T2 = self.padding_image_T2[new_i - self.padding: new_i + 1 + self.padding,
                 new_j - self.padding: new_j + self.padding + 1, :].transpose(2, 0, 1).reshape(self.channel,
                                                                                               self.padding * 2 + 1,
                                                                                               self.padding * 2 + 1)
            label = label
            data_pair = {}
            data_pair['T1'] = T1
            data_pair['T2'] = T2
            return {'data_pair': data_pair,
                    'label': label}
