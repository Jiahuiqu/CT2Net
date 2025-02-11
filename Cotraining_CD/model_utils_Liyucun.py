import torch
from torch import nn
import models
from util.data import data_process as dp
import numpy as np
from collections import OrderedDict
from loss import SoftCrossEntropyLoss
from util.serialization import load_checkpoint, save_checkpoint
import os


def get_optim_params(config, params):

    if config.loss_name not in ['softmax', 'weight_softmax']:
        raise ValueError('wrong loss name')
    optimizer = torch.optim.SGD(params,
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay,
                                nesterov=True)
    if config.loss_name == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = SoftCrossEntropyLoss()
    return criterion, optimizer


def train_model(model, dataloader, config, device):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training config
        criterion
    """
    param_groups = model.parameters()
    criterion, optimizer = get_optim_params(config, param_groups)
    criterion = criterion.to(device)

    def adjust_lr(epoch, step_size=20):
        lr = 0.0001
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    for epoch in range(config.epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        adjust_lr(epoch, config.step_size)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, batch in enumerate(dataloader):
            data_pair = batch['data_pair']
            inputs_T1, inputs_T2 = data_pair['T1'].type(torch.float32).to(device), data_pair['T2'].type(torch.float32).to(device)
            targets = batch['label'].type(torch.float32).to(device)
            weights = batch['weight'].type(torch.float32).to(device)
            outputs = model(inputs_T1, inputs_T2)
            loss = criterion(outputs, targets.long(), weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1) # 预测出的每类的标签
            total += targets.size(0) # 已经训练过的samples的数量，按bz往上加
            correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
              (train_loss /
               (batch_idx + 1), 100. * correct / total, correct, total))

def train_model_rgb(model, dataloader, config, device):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training config
        criterion
    """
    param_groups = model.parameters()
    criterion, optimizer = get_optim_params(config, param_groups)
    criterion = criterion.to(device)


    def adjust_lr(epoch, step_size=20):
        lr = 0.0001
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    best_loss = 1000
    for epoch in range(config.epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        adjust_lr(epoch, config.step_size)
        train_loss = 0
        correct = 0
        total = 0
        diff_features = []
        for batch_idx, batch in enumerate(dataloader):
            data_pair = batch['data_pair']
            inputs_T1, inputs_T2 = data_pair['T1'].type(torch.float32).to(device), data_pair['T2'].type(torch.float32).to(device)
            targets = batch['label'].type(torch.float32).to(device)
            weights = batch['weight'].type(torch.float32).to(device)
            outputs, diff_feature = model(inputs_T1, inputs_T2)
            diff_features.append(diff_feature.cpu().detach().numpy())
            loss = criterion(outputs, targets.long(), weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1) # 预测出的每类的标签
            total += targets.size(0) # 已经训练过的samples的数量，按bz往上加
            correct += predicted.eq(targets).sum().item()

        if np.isnan(train_loss):
            print('loss is nan')
            break

        if best_loss > train_loss:
            best_loss = train_loss
            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'best_loss': train_loss
                },
                False,
                fpath=os.path.join(
                    'spaco/Liyucun/rgb_best'))

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
              (train_loss /
               (batch_idx + 1), 100. * correct / total, correct, total))
    if not os.path.exists('diff_features/Liyucun'):
        os.makedirs('diff_features/Liyucun')
    np.save('diff_features/Liyucun/diff_features.npy', diff_features)

def train_model_hsi(model, dataloader, config, device):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training config
        criterion
    """
    param_groups = model.parameters()
    criterion, optimizer = get_optim_params(config, param_groups)
    criterion = criterion.to(device)
    diff_features = torch.tensor(np.load('diff_features/Liyucun/diff_features.npy')).to(device)

    def adjust_lr(epoch, step_size=20):
        lr = 0.0001
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    best_loss = 1000
    for epoch in range(config.epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        adjust_lr(epoch, config.step_size)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, batch in enumerate(dataloader):
            data_pair = batch['data_pair']
            inputs_T1, inputs_T2 = data_pair['T1'].type(torch.float32).to(device), data_pair['T2'].type(torch.float32).to(device)
            targets = batch['label'].type(torch.float32).to(device)
            weights = batch['weight'].type(torch.float32).to(device)
            bz = targets.size(0)
            diff_feature = diff_features[batch_idx].clone().detach().type(torch.float32).to(device)
            outputs = model(inputs_T1, inputs_T2, diff_feature)
            loss = criterion(outputs, targets.long(), weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1) # 预测出的每类的标签
            total += bz # 已经训练过的samples的数量，按bz往上加
            correct += predicted.eq(targets).sum().item()

        if np.isnan(train_loss):
            print('loss is nan')
            break

        if best_loss > train_loss:
            best_loss = train_loss
            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'best_loss': train_loss
                },
                False,
                fpath=os.path.join(
                    'spaco/Liyucun/HSI_best'))

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
              (train_loss /
               (batch_idx + 1), 100. * correct / total, correct, total))

def train(modality, model, train_data, config, device):
    #  model = models.create(config.model_name)
    #  model = nn.DataParallel(model).cuda()
    dataloader = dp.get_dataloader(train_data, config, is_training=True, mode='train')
    #train_model(model, dataloader, config, device)
    if modality == 'RGB':
        train_model_rgb(model, dataloader, config, device)
    if modality == 'HSI':
        train_model_hsi(model, dataloader, config, device)

    #  return model


def predict_prob(modality, type, model, data, config, device):
    model.eval()

    checkpoint = torch.load('spaco/Liyucun/%s_best' % (modality))
    model.load_state_dict(checkpoint['state_dict'])

    dataloader = dp.get_dataloader(data, config, mode='test')
    probs = []
    diff_features = []

    if modality == 'RGB':
        diff_features = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data_pair = batch['data_pair']
                inputs_T1, inputs_T2 = data_pair['T1'].type(torch.float32).to(device), data_pair['T2'].type(
                    torch.float32).to(device)
                targets = batch['label'].type(torch.float32).to(device)
                output, diff_feature = model(inputs_T1, inputs_T2)
                diff_features.append(diff_feature.cpu().detach().numpy())
                prob = nn.functional.softmax(output, dim=1)
                probs += [prob.data.cpu().numpy()]
        np.save('diff_features/Liyucun/diff_features_%s.npy' % (type), diff_features)
    else:
        diff_features = np.load('diff_features/Liyucun/diff_features_%s.npy' % (type), allow_pickle=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data_pair = batch['data_pair']
                inputs_T1, inputs_T2 = data_pair['T1'].type(torch.float32).to(device), data_pair['T2'].type(
                    torch.float32).to(device)
                targets = batch['label'].type(torch.float32).to(device)
                diff_feature = torch.tensor(diff_features[batch_idx]).clone().detach().type(torch.float32).to(device)
                output = model(inputs_T1, inputs_T2, diff_feature)
                prob = nn.functional.softmax(output, dim=1)
                probs += [prob.data.cpu().numpy()]
    return np.concatenate(probs)


def evaluate(modality, type, model, data, config, device):
    model.eval()
    checkpoint = torch.load('spaco/Liyucun/%s_best' % (modality))
    model.load_state_dict(checkpoint['state_dict'])
    print("min_loss:", checkpoint['best_loss'])
    correct = 0
    total = 0
    dataloader = dp.get_dataloader(data, config, mode='test')
    diff_features = []

    if modality == 'RGB':
        diff_features = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data_pair = batch['data_pair']
                inputs_T1, inputs_T2 = data_pair['T1'].type(torch.float32).to(device), data_pair['T2'].type(
                    torch.float32).to(device)
                targets = batch['label'].type(torch.float32).to(device)
                outputs, diff_feature = model(inputs_T1, inputs_T2)
                diff_features.append(diff_feature.cpu().detach().numpy())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        np.save('diff_features/Liyucun/diff_features_%s.npy' % (type), diff_features)
    else:
        diff_features = np.load('diff_features/Liyucun/diff_features_%s.npy' % (type), allow_pickle=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data_pair = batch['data_pair']
                inputs_T1, inputs_T2 = data_pair['T1'].type(torch.float32).to(device), data_pair['T2'].type(
                    torch.float32).to(device)
                targets = batch['label'].type(torch.float32).to(device)
                diff_feature = torch.tensor(diff_features[batch_idx]).clone().detach().type(torch.float32).to(device)
                outputs = model(inputs_T1, inputs_T2, diff_feature)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print('Accuracy on Test data: %0.5f' % acc)
    return acc
