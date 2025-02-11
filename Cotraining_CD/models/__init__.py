from .hsi_bay import *
from .hsi_bay_woAACo import *
from .rgb_bay import *
from .hsi_Liyucun import *
from .hsi_Liyucun_woAACo import *
from .rgb_Liyucun import *
from .hsi_bar import *
from .rgb_bar import *
from .hsi_china import *
from .rgb_china import *
from .hsi_usa import *
from .rgb_usa import *

#在这里加模型名字和结构
__factory = {
    'hsi_bay': BCNN_hsi_bay,
    'hsi_bay_woAACo': BCNN_hsi_bay_woAACo,
    'rgb_bay': BCNN_rgb_bay,
    'hsi_Liyucun': BCNN_hsi_Liyucun,
    'hsi_Liyucun_woAACo': BCNN_hsi_Liyucun_woAACo,
    'rgb_Liyucun': BCNN_rgb_Liyucun,
    'hsi_bar': BCNN_hsi_bar,
    'rgb_bar': BCNN_rgb_bar,
    'hsi_china': BCNN_hsi_china,
    'rgb_china': BCNN_rgb_china,
    'hsi_usa': BCNN_hsi_usa,
    'rgb_usa': BCNN_rgb_usa
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.
    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)

