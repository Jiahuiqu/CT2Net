import numpy as np
from util.data import transforms as T
from torch.utils.data import DataLoader
from .preprocessor import Preprocessor


def get_augmentation_func_list(aug_list, config):
    if aug_list is None: return []
    assert isinstance(aug_list, list)
    aug_func = []
    for aug in aug_list:
        if aug == 'rf':
            aug_func += [T.RandomHorizontalFlip()]
        elif aug == 'rc':
            aug_func += [T.RandomCrop(config.height, padding=config.padding)]
        elif aug == 're':
            aug_func += [T.RandomErasing(probability=0.5, sh=0.4, r1=0.3)]
        else:
            raise ValueError('wrong augmentation name')
    return aug_func



def get_transformer(config, is_training=False):
    normalizer = T.Normalize(mean=config.mean, std=config.std)
    base_transformer = [T.ToTensor(), normalizer]
    if not is_training:
        return T.Compose(base_transformer)
    aug1 = T.RandomErasing(probability=0.5, sh=0.4, r1=0.3)
    early_aug = get_augmentation_func_list(config.early_transform, config)
    later_aug = get_augmentation_func_list(config.later_transform, config)
    aug_list = early_aug + base_transformer + later_aug
    return T.Compose(aug_list)


def get_dataloader(dataset, config, is_training=False, mode='train'):
    # transformer = get_transformer(config, is_training=is_training)
    sampler = None
    if is_training and config.sampler:
        sampler = config.sampler(dataset, config.num_instances)
    data_loader = DataLoader(Preprocessor(dataset, mode),
                             batch_size=config.batch_size,
                             num_workers=config.workers,
                             shuffle=is_training,
                             sampler=sampler,
                             pin_memory=True,
                             drop_last=is_training)
    return data_loader


def update_train_untrain(sel_idx,
                         train_data,
                         untrain_data,
                         pred_y,
                         weights=None):
    #  assert len(train_data) == len(untrain_data)
    if weights is None:
        weights = np.ones(len(untrain_data[1]), dtype=np.float32) # untrain_data的数量
    add_data = [
        untrain_data[0], np.array(untrain_data[1])[sel_idx],np.array(pred_y)[sel_idx], train_data[3], train_data[4], weights[sel_idx]
    ]
    new_untrain = [
      untrain_data[0], np.array(untrain_data[1])[~sel_idx],np.array(pred_y)[~sel_idx], untrain_data[3], untrain_data[4], weights[~sel_idx]
    ]
    new_train = [
      train_data[0], np.concatenate((train_data[1], add_data[1])), np.concatenate((train_data[2], add_data[2])), train_data[3], train_data[4], np.concatenate((train_data[5], add_data[5]))
    ]
    return new_train, new_untrain




def select_ids(score, train_data, max_add):
    y = train_data[1]
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    pred_y = np.argmax(score, axis=1)
    ratio_per_class = [sum(y == c)/len(y) for c in clss]
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(ratio_per_class[cls] * max_add)),
                      indices.shape[0])
        add_indices[indices[idx_sort[-add_num:]]] = 1
    return add_indices.astype('bool')


def get_lambda_class(score, pred_y, train_data, max_add):
    y = train_data[2]
    lambdas = np.zeros(score.shape[1]) # arr(2,)
    add_ids = np.zeros(score.shape[0]) # arr(num(untrain_data),)
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    ratio_per_class = np.full((len(clss),), 1/len(clss))

    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        if len(indices) == 0:
            continue
        cls_score = score[indices, cls] # 取出某一类别中每个置信度
        idx_sort = np.argsort(cls_score) # 按置信度由小到大排列的索引值
        add_num = min(int(np.ceil(ratio_per_class[cls] * max_add)),
                      indices.shape[0])
        add_ids[indices[idx_sort[-add_num:]]] = 1 # 选出置信度最高的add_num个样本
        lambdas[cls] = cls_score[idx_sort[-add_num]] - 0.1 # 每类置信度最大值减0.1
    return add_ids.astype('bool'), lambdas


def get_ids_weights(pred_prob, pred_y, train_data, max_add, gamma, regularizer='hard'):
    '''
    pred_prob: predicted probability of all views on untrain data
    pred_y: predicted label for untrain data
    train_data: training data
    max_add: number of selected data
    gamma: view correlation hyper-parameter
    '''

    add_ids, lambdas = get_lambda_class(pred_prob, pred_y, train_data, max_add)
    weight = np.array([(pred_prob[i, l] - lambdas[l]) / (gamma + 1e-5)
                       for i, l in enumerate(pred_y)], # 权重更新公式 pred_prob[i, l]指untrain_data中每个样本的预测最大概率值
                      dtype='float32')
    weight[~add_ids] = 0
    if regularizer == 'hard' or gamma == 0:
        weight[add_ids] = 1
        return add_ids, weight
    weight[weight < 0] = 0
    weight[weight > 1] = 1
    return add_ids, weight

def update_ids_weights(view, probs, sel_ids, weights, pred_y, train_data,
                       max_add, gamma, regularizer='hard'):
    num_view = len(probs)
    for v in range(num_view):
        if v == view:
            continue
        ov = sel_ids[v]

        probs[view][ov, pred_y[ov]] += gamma * weights[v][ov] / (num_view - 1) # 在两个view处的权重都为1的位置，将probs[view]中weight为1的样本上的最大预测概率加gamma
    sel_id, weight = get_ids_weights(probs[view], pred_y, train_data,
                                     max_add, gamma, regularizer)
    return sel_id, weight

def get_weights(pred_prob, pred_y, train_data, max_add, gamma, regularizer):
    lamb = get_lambda_class(pred_prob, pred_y, train_data, max_add)
    weight = np.array([(pred_prob[i, l] - lamb[l]) / gamma
                       for i, l in enumerate(pred_y)],
                      dtype='float32')
    if regularizer is 'hard':
        weight[weight > 0] = 1
        return weight
    weight[weight > 1] = 1
    return weight


def split_dataset(dataset, train_ratio=0.5, seed=0, num_per_class=400):
    """
    split dataset to train_set and untrain_set
    """
    assert 0 <= train_ratio <= 1
    np.random.seed(seed)
    pid = np.squeeze(dataset[2].reshape(1, -1)) # REF
    pids = pid[dataset[1]] # train_data对应的REF
    clss = np.unique(pids)
    sel_ids = np.zeros(len(dataset[1]), dtype=bool)
    for cls in clss:
        indices = np.where(pids == cls)[0]
        np.random.shuffle(indices)
        if num_per_class:
            sel_id = indices[:num_per_class]# 每类被选择出的标签id
        else:
            train_num = int(np.ceil((len(indices) * train_ratio)))
            sel_id = indices[:train_num]
        sel_ids[sel_id] = True
    train_set = []
    untrain_set = []
    train_idx = []
    untrain_idx = []
    train_REF = []
    untrain_REF = []
    train_set.append(dataset[0])
    untrain_set.append(dataset[0])
    for i in range(len(sel_ids)):
        if sel_ids[i]:
            train_idx.append(dataset[1][i])
            train_REF.append(pid[dataset[1][i]])
        else:
            untrain_idx.append(dataset[1][i])
            untrain_REF.append(pid[dataset[1][i]])
    train_set.append(np.array(train_idx))# 被选择的训练集，每类根据train_ratio选择
    untrain_set.append(np.array(untrain_idx)) # 未被选择的训练集
    train_set.append(np.array(train_REF))  # 被选择的训练集的REF
    untrain_set.append(np.array(untrain_REF))  # 未被选择的训练集的REF
    train_set.append(dataset[3])
    untrain_set.append(dataset[3])
    train_set.append(dataset[4])
    untrain_set.append(dataset[4])
    ### add sample weight
    train_set += [np.full((len(train_set[1])), 1.0)]
    return train_set, untrain_set, sel_ids

def split_dataset_idx(dataset, index):
    """
    using sel_ids to split dataset to train_set and untrain_set
    """
    pid = np.squeeze(dataset[2].reshape(1, -1)) # REF
    train_set = []
    untrain_set = []
    train_idx = []
    untrain_idx = []
    train_REF = []
    untrain_REF = []
    train_set.append(dataset[0])
    untrain_set.append(dataset[0])
    for i in range(len(index)):
        if index[i]:
            train_idx.append(dataset[1][i])
            train_REF.append(pid[dataset[1][i]])
        else:
            untrain_idx.append(dataset[1][i])
            untrain_REF.append(pid[dataset[1][i]])
    train_set.append(train_idx)  # 被选择的训练集，每类根据train_ratio选择
    untrain_set.append(untrain_idx)  # 未被选择的训练集
    train_set.append(train_REF)  # 被选择的训练集的REF
    untrain_set.append(untrain_REF)  # 未被选择的训练集的REF
    train_set.append(dataset[3])
    untrain_set.append(dataset[3])
    train_set.append(dataset[4])
    untrain_set.append(dataset[4])
    ### add sample weight
    train_set += [np.full((len(train_set[1])), 1.0)]
    return train_set, untrain_set