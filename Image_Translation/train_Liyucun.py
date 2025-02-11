import numpy as np
from torch.utils.data import DataLoader
import itertools
from utils import LambdaLR
import torch
from utils import weights_init_normal
from torch.autograd import Variable
import warnings
from Data_Liyucun import Dataset
from model_Liyucun import guided_SpeSRModel, guided_SpaSRModel, Discriminator
import matplotlib.pyplot as plt
import os

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
device = torch.device("cuda:1")

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    lr = 0.0001
    max_epoch = 20  # All  train epoch
    decay_epoch = 5  # epoch to start linearly decaying the learning rate to 0
    batchSize = 1

    # Dataset loader


    Cin1 = 194
    Cin2 = 3
    path = '/run/media/xd132/E/RJY/1.first/datasets/Liyucun'

    db = Dataset(path, 'T1HSI.mat', 'T2RGB.mat', 'REF.mat', 'train', 194, 3)
    train_data = DataLoader(db, batch_size=batchSize, shuffle=True)
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
    netD_rgb = Discriminator(3)
    netD_hsi = Discriminator(194)
    ###
    netG_rgb2hsi = netG_rgb2hsi.to(device)
    netG_hsi2rgb = netG_hsi2rgb.to(device)
    netD_rgb = netD_rgb.to(device)
    netD_hsi = netD_hsi.to(device)
    ###
    # netG_rgb2hsi.apply(weights_init_normal)
    # netG_hsi2rgb.apply(weights_init_normal)
    # netD_rgb.apply(weights_init_normal)
    # netD_hsi.apply(weights_init_normal)

    ###loss function
    criterion_public = torch.nn.L1Loss()
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    loss_l1 = torch.nn.L1Loss()
    # crit_vgg = perceptual_loss.VGGLoss()
    ###
    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_rgb2hsi.parameters(), netG_hsi2rgb.parameters()),
                                   lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_rgb.parameters(), lr=lr*5, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_hsi.parameters(), lr=lr*5, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(max_epoch, 0, decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(max_epoch, 0, decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(max_epoch, 0, decay_epoch).step)

    # Use 0.9 to avoid training the discriminators to zero loss
    target_real = Variable(torch.Tensor(batchSize).fill_(1.0)*0.9, requires_grad=False).to(device)
    target_fake = Variable(torch.Tensor(batchSize).fill_(0.0), requires_grad=False).to(device)

    ###################################
    if not os.path.exists('output/Liyucun'):
        os.makedirs('output/Liyucun')
    # state_dict = torch.load('output/Liyucun/15netG_hsi2rgb.pth')
    # netG_hsi2rgb.load_state_dict(state_dict)
    # state_dict = torch.load('output/Liyucun/15netG_rgb2hsi.pth')
    # netG_rgb2hsi.load_state_dict(state_dict)
    # state_dict = torch.load('output/Liyucun/15netD_hsi.pth')
    # netD_hsi.load_state_dict(state_dict)
    # state_dict = torch.load('output/Liyucun/15netD_rgb.pth')
    # netD_rgb.load_state_dict(state_dict)

    ##### Training ######
    for epoch in range(max_epoch):
        for i, (hsi, rgb, label) in enumerate(train_data):
            # Set model input
            real_rgb = rgb.type(torch.float).to(device)
            real_hsi = hsi.type(torch.float).to(device)

            ###### Generators A2B and B2A ####+##
            syn_hsi = netG_rgb2hsi(real_rgb, real_hsi)
            syn_rgb = netG_hsi2rgb(real_hsi, real_rgb)

            # L1 loss of central pixel
            hsi_index = int(np.ceil(size_hsi / 2))
            rgb_index = int(np.ceil(size_rgb / 2))
            loss_hsi = loss_l1(syn_hsi[:, :, hsi_index, hsi_index], real_hsi[:, :, hsi_index, hsi_index])# 计算rgb2hsi与real_hsi中心像素之间的L1loss
            loss_rgb = loss_l1(syn_rgb[:, :, rgb_index - 1:rgb_index + 2, rgb_index - 1:rgb_index + 2],
                               real_rgb[:, :, rgb_index - 1:rgb_index + 2, rgb_index - 1:rgb_index + 2])  # 计算hsi2rgb与real_rgb之间的L1loss

            # # perceptual loss
            # loss_vgg_rgb = VGG_rgb(real_rgb, syn_rgb)
            # loss_vgg_hsi = VGG_hsi(real_hsi, syn_hsi, Cin1)

            # GAN loss
            pred_syn_hsi = netD_hsi(syn_hsi)
            loss_GAN_rgb2hsi = criterion_GAN(pred_syn_hsi, target_real)

            pred_syn_rgb = netD_rgb(syn_rgb)
            loss_GAN_hsi2rgb = criterion_GAN(pred_syn_rgb, target_real)

            # Cycle loss
            recovered_rgb = netG_hsi2rgb(syn_hsi, syn_rgb)
            # recovered_rgb = netG_hsi2rgb(syn_hsi)
            loss_cycle_ABA = criterion_cycle(recovered_rgb, real_rgb)

            recovered_hsi = netG_rgb2hsi(syn_rgb, syn_hsi)
            # recovered_hsi = netG_rgb2hsi(syn_rgb)
            loss_cycle_BAB = criterion_cycle(recovered_hsi, real_hsi)

            # total loss
            loss_G = loss_hsi + loss_rgb + 10*(loss_cycle_ABA + loss_cycle_BAB) + loss_GAN_rgb2hsi + loss_GAN_hsi2rgb
            ###################################

            ###### Discriminator Training A&B ######
            # discriminator should predicts all patches of real images as real (1)
            # Real loss
            pred_real_rgb = netD_rgb(real_rgb)
            loss_D_real_rgb = criterion_GAN(pred_real_rgb, target_real)
            pred_real_hsi = netD_hsi(real_hsi)
            loss_D_real_hsi = criterion_GAN(pred_real_hsi, target_real)

            # discriminator should predicts all patches of synthetic images as fake (0)
            # Fake loss
            pred_fake_rgb = netD_rgb(syn_rgb.detach())
            loss_D_fake_rgb = criterion_GAN(pred_fake_rgb, target_fake)

            pred_fake_hsi = netD_hsi(syn_hsi.detach())
            loss_D_fake_hsi = criterion_GAN(pred_fake_hsi, target_fake)

            # Total loss
            loss_D_A = (loss_D_real_rgb + loss_D_fake_rgb) * 0.5
            loss_D_B = (loss_D_real_hsi + loss_D_fake_hsi) * 0.5

            if i % 3 == 0:
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                optimizer_D_A.zero_grad()
                loss_D_A.backward()
                optimizer_D_A.step()

                optimizer_D_B.zero_grad()
                loss_D_B.backward()
                optimizer_D_B.step()

            else:
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            ###################################
            frgb = syn_rgb.permute(0,2,3,1).detach().cpu().numpy()
            lrgb = real_rgb.permute(0,2,3,1).detach().cpu().numpy()
            fhsi = syn_hsi.permute(0,2,3,1).detach().cpu().numpy()
            lhsi = real_hsi.permute(0,2,3,1).detach().cpu().numpy()

            if (epoch % 1 == 0) & (i == 0):
            # if epoch % 1 == 0:
                print("epoch:{} step:{} ||Gan_loss--> rgb2hsi:{}|| hsi2rgb:{}".format(epoch, i, loss_hsi, loss_rgb))
                print("cycle_hsi2rgb:{} || cycle_rgb2hsi:{}".format(loss_cycle_ABA, loss_cycle_BAB))
                print("loss_D_A:{} || loss_D_B:{}\n".format(loss_D_A, loss_D_B))

                plt.figure()

                plt.subplot(2, 2, 1)
                plt.imshow(frgb[0, :, :, :])

                plt.subplot(2, 2, 2)
                plt.imshow(lrgb[0, :, :, :])

                plt.subplot(2, 2, 3)

                plt.imshow(fhsi[0, :, :, [109,70,36]].transpose(1, 2, 0))

                plt.subplot(2, 2, 4)
                plt.imshow(lhsi[0, :, :, [109,70,36]].transpose(1, 2, 0))
                plt.title('Liyucun_model_orignal%d' % (epoch))
                plt.show()
                print(i)

        # if loss_hsi <= best_loss:
        print('save_success')
        best_loss = loss_hsi
        # Save models checkpoints
        torch.save(netG_hsi2rgb.state_dict(), 'output/Liyucun/%dnetG_hsi2rgb.pth' % (epoch))
        torch.save(netG_rgb2hsi.state_dict(), 'output/Liyucun/%dnetG_rgb2hsi.pth' % (epoch))
        torch.save(netD_hsi.state_dict(), 'output/Liyucun/%dnetD_hsi.pth' % (epoch))
        torch.save(netD_rgb.state_dict(), 'output/Liyucun/%dnetD_rgb.pth' % (epoch))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()


