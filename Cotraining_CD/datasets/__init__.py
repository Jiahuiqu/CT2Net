from __future__ import absolute_import
import numpy as np
import warnings
import torchvision
from dataset_hsi import Dataset_hsi
from dataset_rgb import Dataset_rgb


__factory = ['hsi', 'rgb']



def create(modality, root, T1, T2, GT, dataset_name, channel, download=True, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'cifar10', 'mnist', 'cifar100'
    root : str
        The path to the dataset directory.
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if modality == 'hsi':
        data = {}
        trainset = Dataset_hsi(root+T1, root+T2, root+GT, dataset_name, 'train', channel)
        data['train'] = [trainset.data_pair, np.array(trainset.random_point), np.array(trainset.image_REF), trainset.channel, trainset.padding]
        testset = Dataset_hsi(root+T1, root+T2, root+GT, dataset_name, 'test', channel)
        data['test'] = [testset.data_pair, np.array(testset.random_point), np.array(testset.whole_REF), testset.channel, testset.padding]
        return  data
    elif modality == 'rgb':
        data = {}
        trainset = Dataset_rgb(root + T1, root + T2, root + GT, dataset_name, 'train', channel)
        data['train'] = [trainset.data_pair, np.array(trainset.random_point), np.array(trainset.image_REF), trainset.channel, trainset.padding]
        testset = Dataset_rgb(root + T1, root + T2, root + GT, dataset_name, 'test', channel)
        data['test'] = [testset.data_pair, np.array(testset.random_point), np.array(testset.whole_REF), testset.channel, testset.padding]
        return  data
    else:
        raise KeyError("Unknown dataset:", modality)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)


if __name__ == '__main__':
    data_dir = '/run/media/xd132/E/RJY/1.first/Image_Translation_CycleGAN/result/data/'
    T1HSI = 'T1HSI_upsample.mat'
    T2HSI = 'T2HSI_upsample_modify_hsi_2.mat'
    T1RGB = 'T1RGB_modify_rgb.mat'
    T2RGB = 'T2RGB.mat'
    GT = 'REF.mat'
    dataset_name = 'Bar'
    hsi_channel = 224
    rgb_channel = 3

    modality = 'hsi'
    data_hsi = create(modality, data_dir, T1HSI, T2HSI, GT, dataset_name, hsi_channel)
    modality = 'rgb'
    data_rgb = create(modality, data_dir, T1RGB, T2RGB, GT, dataset_name, rgb_channel)