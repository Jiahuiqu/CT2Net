import os
import torch
import argparse
import model_utils_Liyucun as mu
from util.data import data_process as dp
from config import Config
from util.serialization import load_checkpoint, save_checkpoint
import datasets
import models
import numpy as np
import torch.multiprocessing as mp
from scipy.io import savemat

device = torch.device("cuda:1")
if not os.path.exists('spaco/Liyucun'):
    os.makedirs('spaco/Liyucun')

parser = argparse.ArgumentParser(description='soft_spaco')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-r', '--regularizer', type=str, default='hard')
parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('--gamma', type=float, default=0.3)
parser.add_argument('--iter-steps', type=int, default=3)
parser.add_argument('--num-per-class', type=int, default=0) # 400
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def train_predict(net, train_data, untrain_data, test_data, config, device, pred_probs):
    mu.train(net, train_data, config, device)
    pred_probs.append(mu.predict_prob(net, untrain_data, configs[view], view))


def parallel_train(nets, train_data, data_dir, configs):
    processes = []
    for view, net in enumerate(nets):
        p = mp.Process(target=mu.train, args=(net, train_data, config, view))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()




def adjust_config(config, num_examples, iter_step):
    repeat = 20 * (1.1 ** iter_step)

    # if config.model_name == 'rgb':
    #     config.epochs = 80  # 300
    # elif config.model_name == 'hsi':
    #     config.epochs = 300  # 300
    config.epochs = 300

    config.step_size = max(int(config.epochs // 3), 1)
    return config
def spaco(configs,
          data,
          iter_steps=1,
          gamma=0,
          train_ratio=0.2,
          regularizer='soft'):
    """
    self-paced co-training model implementation based on Pytroch
    params:
    model_names: model names for spaco, such as ['resnet50','densenet121']
    data: dataset for spaco model
    save_pathts: save paths for two models
    iter_step: iteration round for spaco
    gamma: spaco hyperparameter
    train_ratio: initiate training dataset ratio
    """
    num_view = len(configs)
    data_rgb = data[0]
    data_hsi = data[1]
    # 在训练数据里分出train_data和untrain_data，选择训练集的比例为0.5
    train_data_hsi, untrain_data_hsi, sel_ids = dp.split_dataset(
        data_hsi['train'], seed=args.seed, num_per_class=args.num_per_class)
    train_data_rgb, untrain_data_rgb = dp.split_dataset_idx(data_rgb['train'], sel_ids)
    train_data = [train_data_rgb, train_data_hsi]
    untrain_data = [untrain_data_rgb, untrain_data_hsi]
    test_data = [data_rgb['test'], data_hsi['test']]
    add_num = 250
    pred_probs = []
    test_preds = []
    sel_ids = []
    weights = []
    start_step = 0
    ###########
    # initiate classifier to get preidctions
    ###########
    for view in range(num_view):
    # for view in range(1, 2):
        configs[view] = adjust_config(configs[view], len(train_data[view][1]), 0)
        print(configs[view].model_name)
        net = models.create(configs[view].model_name).to(device)
        ###RGB###
        if view == 0:
            mu.train('RGB', net, train_data[view], configs[view], device=device) # 用train_data初始化网络
            pred_probs.append(mu.predict_prob('RGB', 'untrain', net, untrain_data[view], configs[view], device))
            test_preds.append(mu.predict_prob('RGB', 'test', net, test_data[view], configs[view], device))  # 每个view对test的预测标签
        else:
            mu.train('HSI', net, train_data[view], configs[view], device=device)
            pred_probs.append(mu.predict_prob('HSI', 'untrain', net, untrain_data[view], configs[view], device))
            test_preds.append(mu.predict_prob('HSI', 'test', net, test_data[view], configs[view], device))  # 每个view对test的预测标签
    # 每个view对untrain_data的预测标签shape为(num(untrain_data),10)), 在加入view2后，维度变为{list:2}pre_probs[0]和[1]都是(6000, 10)，代表每个view的预测概率
        # acc = mu.evaluate(net, test_data[view], configs[view], device)
        save_checkpoint(
          {
            'state_dict': net.state_dict(),
            'epoch': 0,
          },
          False,
          fpath=os.path.join(
            'spaco/Liyucun/%s.epoch%d' % (configs[view].model_name, 0)))
    pred_y = np.argmax(sum(pred_probs), axis=1) # 两个view的预测概率相加得到融合后的预测标签

    # initiate weights for unlabled examples
    for view in range(num_view):
        # 传traindata进去是为了得到label的类别数量
        sel_id, weight = dp.get_ids_weights(pred_probs[view], pred_y,
                                            train_data[view], add_num, gamma,
                                            regularizer)
        # import pdb;pdb.set_trace()
        sel_ids.append(sel_id) # {list:2} untraindata的数量
        weights.append(weight) # 被选择的标签处权重为1

    # start iterative training
    gt_y = test_data[0][2] # whole_REF
    # gt_y = np.squeeze(gt_y.reshape(1, -1))
    for step in range(start_step, iter_steps):
        for view in range(num_view):
            print('Iter step: %d, view: %d, model name: %s' % (step+1,view,configs[view].model_name))

            # update sample weights
            sel_ids[view], weights[view] = dp.update_ids_weights(
              view, pred_probs, sel_ids, weights, pred_y, train_data[view],
              add_num, gamma, regularizer)
            # update model parameter
            new_train_data, _ = dp.update_train_untrain(
              sel_ids[view], train_data[view], untrain_data[view], pred_y, weights[view])
            configs[view] = adjust_config(configs[view], len(train_data[view][1]), step)
            net = models.create(configs[view].model_name).to(device)
            ###RGB###
            if view == 0:
                mu.train('RGB', net, new_train_data, configs[view], device=device)

                # update y
                pred_probs[view] = mu.predict_prob('RGB', 'untrain', net, untrain_data[view],
                                                   configs[view], device)

                # evaluation current model and save it
                acc = mu.evaluate('RGB', 'eval', net, test_data[view], configs[view], device=device)
                predictions = mu.predict_prob('RGB', 'train', net, train_data[view], configs[view], device=device)
                test_preds[view] = mu.predict_prob('RGB', 'test', net, test_data[view], configs[view], device=device)
            else:
                mu.train('HSI', net, new_train_data, configs[view], device=device)

                # update y
                pred_probs[view] = mu.predict_prob('HSI', 'untrain', net, untrain_data[view],
                                                   configs[view], device)

                # evaluation current model and save it
                acc = mu.evaluate('HSI', 'eval', net, test_data[view], configs[view], device=device)
                predictions = mu.predict_prob('HSI', 'train', net, train_data[view], configs[view], device=device)
                test_preds[view] = mu.predict_prob('HSI', 'test', net, test_data[view], configs[view], device=device)
            save_checkpoint(
              {
                'state_dict': net.state_dict(),
                'epoch': step + 1,
                'predictions': predictions,
                'accuracy': acc
              },
              False,
              fpath=os.path.join(
                'spaco/Liyucun/%s.epoch%d' % (configs[view].model_name, step + 1)))

        add_num += 40 * num_view # 源码写的4000
        fuse_y = np.argmax(sum(test_preds), axis=1)
        outimage = np.zeros((1, h*w))
        outimage = fuse_y.reshape(h, w)
        if not os.path.exists('result/Liyucun'):
            os.makedirs('result/Liyucun')
        savemat('result/Liyucun/Liyucun_%d.mat' % (step), {"data": outimage})
        print('Acc:%0.4f' % np.mean(fuse_y == gt_y))
    #  print(acc)


config1 = Config(model_name='rgb_Liyucun', loss_name='weight_softmax')
config2 = Config(model_name='hsi_Liyucun', loss_name='weight_softmax')


cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = '/run/media/xd132/E/RJY/1.first/datasets/Liyucun/'
T1HSI = 'T1HSI_upsample.mat'
T2HSI = 'T2HSI_upsample.mat'
T1RGB = 'T1RGB.mat'
T2RGB = 'T2RGB.mat'
GT = 'REF.mat'
dataset_name = 'Liyucun'
hsi_channel = 194
rgb_channel = 3
h = 750
w = 375


modality = 'hsi'
data_hsi = datasets.create(modality, data_dir, T1HSI, T2HSI, GT, dataset_name, hsi_channel)
modality = 'rgb'
data_rgb = datasets.create(modality, data_dir, T1RGB, T2RGB, GT, dataset_name, rgb_channel)

spaco([config1, config2],
      [data_rgb, data_hsi],
      gamma=args.gamma,
      iter_steps=args.iter_steps,
      regularizer=args.regularizer)


# if __name__ == '__main__':
#     model = models.create('BCNN_hsi')