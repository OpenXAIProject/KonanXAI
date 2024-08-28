from .dataset import Datasets
from .coco import COCO
from .custom import CUSTOM
from .mnist import MNIST
from .cifar10 import CIFAR10
from .ai_fire import AI_FIRE
__all__ = ["Datasets", "CUSTOM","MNIST", "COCO", "CIFAR10","AI_FIRE"]


def load_dataset(framework, data_path = None, data_type = 'CUSTOM', 
                 maxlen=-1, resize = None):
    dataset = globals().get(data_type)(framework = framework, src_path = data_path)
    dataset.set_fit_size(resize)
    return dataset