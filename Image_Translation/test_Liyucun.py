import numpy as np
from torch.utils.data import DataLoader
import itertools
from utils import LambdaLR
import torch
import warnings
from Data_Liyucun import Dataset
from model_Liyucun import guided_SpeSRModel, guided_SpaSRModel
import matplotlib.pyplot as plt
import os
import cv2 as cv
from scipy.io import savemat, loadmat


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
device = torch.device("cuda:0")

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    lr = 0.0001
    max_epoch = 1000  # All  train epoch
    decay_epoch = 50  # epoch to start linearly decaying the learning rate to 0
    batchSize = 1

    # Dataset loader
    Cin1 = 194
    Cin2 = 3
    path = '/run/media/xd132/E/RJY/1.first/datasets/Liyucun'

    db = Dataset(path, 'T1HSI.mat', 'T2RGB.mat', 'REF.mat', 'test', 194, 3)
    test_data = DataLoader(db, batch_size=batchSize, shuffle=False)
    best_loss = 10000
    loss_hsi = 0

    ###### Definition of variables ######
    # Networks
    ###########################
    size_hsi = 27
    size_rgb = size_hsi * 3
    patch_hsi = size_hsi ** 2
    patch_rgb = size_rgb ** 2
    ###########################
    ### init model
    netG_rgb2hsi = guided_SpeSRModel(input_features=3, output_features=194)
    netG_hsi2rgb = guided_SpaSRModel(input_features=194, output_features=3)
    ###
    netG_rgb2hsi = netG_rgb2hsi.to(device)
    netG_hsi2rgb = netG_hsi2rgb.to(device)
    ###

    ###loss function
    criterion_public = torch.nn.L1Loss()
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    loss_l1 = torch.nn.L1Loss()

    ###
    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_rgb2hsi.parameters(), netG_hsi2rgb.parameters()),
                                   lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(max_epoch, 0, decay_epoch).step)

    ###################################
    netG_hsi2rgb.eval()
    state_dict = torch.load('output/Liyucun/19netG_hsi2rgb.pth')
    netG_hsi2rgb.load_state_dict(state_dict)
    netG_rgb2hsi.eval()
    state_dict = torch.load('output/Liyucun/19netG_rgb2hsi.pth')
    netG_rgb2hsi.load_state_dict(state_dict)

    ##### test ######
    if not os.path.exists('result/Liyucun'):
        os.makedirs('result/Liyucun')

    hsi_index = int(np.ceil(size_hsi / 2))
    rgb_index = int(np.ceil(size_rgb / 2))

    outhsi = np.zeros((250, 125, 194))
    padding = hsi_index-1
    padding_outhsi = cv.copyMakeBorder(outhsi, padding, padding, padding, padding,
                                                  cv.BORDER_REFLECT)
    outrgb = np.zeros((750, 375, 3))
    padding = rgb_index-1
    padding_outrgb = cv.copyMakeBorder(outrgb, padding, padding, padding, padding,
                                                  cv.BORDER_REFLECT)

    with torch.no_grad():
        for step, (hsi, rgb, label, hsi_i, hsi_j, rgb_i, rgb_j) in enumerate(test_data):
            real_hsi = hsi.type(torch.float32).to(device)
            real_rgb = rgb.type(torch.float32).to(device)
            syn_hsi = netG_rgb2hsi(real_rgb, real_hsi).permute(0, 2, 3, 1)
            syn_rgb = netG_hsi2rgb(real_hsi, real_rgb).permute(0, 2, 3, 1)

            out_synhsi = syn_hsi[0, hsi_index, hsi_index, :]
            out_synrgb = syn_rgb[0, rgb_index - 1:rgb_index + 2, rgb_index - 1:rgb_index + 2, :]
            out_synhsi = out_synhsi.detach().cpu().numpy()
            out_synrgb = out_synrgb.detach().cpu().numpy()

            padding_outhsi[hsi_i, hsi_j, :] = out_synhsi
            padding_outrgb[rgb_i-1:rgb_i+2, rgb_j-1:rgb_j+2, :] = out_synrgb
        filename = "result/Liyucun/hsi_19.mat"
        savemat(filename, {"hsi": padding_outhsi})
        filename = "result/Liyucun/rgb_19.mat"
        savemat(filename, {"rgb": padding_outrgb})
        print("save success!!!!")

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(padding_outrgb)

        plt.subplot(1, 2, 2)
        plt.imshow(padding_outhsi[:, :, [109,70,36]])
        plt.show()

        image_T1 = loadmat('/run/media/xd132/E/RJY/1.first/datasets/Liyucun/T1HSI.mat')['T2']
        image_T2 = loadmat('/run/media/xd132/E/RJY/1.first/datasets/Liyucun/T2RGB.mat')['T1']

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image_T1[:, :, [109,70,36]])

        plt.subplot(1, 2, 2)
        plt.imshow(image_T2)
        plt.show()