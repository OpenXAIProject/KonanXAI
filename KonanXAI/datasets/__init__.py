from .dataset import Datasets
from .coco import COCO
from .custom import CUSTOM
from .mnist import MNIST
from .cifar10 import CIFAR10
from .ai_fire import AI_FIRE
from .dann_ai_fire import DANN_AI_FIRE
from .counterfactual import Counterfactual
#__all__ = ["Datasets", "CUSTOM","MNIST", "COCO", "CIFAR10","AI_FIRE", "DANN_AI_FIRE"]


def load_dataset(framework, data_path = None, data_type = 'CUSTOM', 
                 maxlen=-1, resize = None, mode = None, label = None):
    dataset = globals().get(data_type)(framework = framework, src_path = data_path)
    dataset.set_fit_size(resize)
    if mode == 'train':
        dataset.set_train()
    elif mode == 'explain':
        dataset.set_test()
    elif mode == 'evaluation':
        dataset.set_test()
    elif mode == 'explainer':
        dataset.set_train()
    elif mode == 'counterfactual':
        dataset.set_train()
        dataset.set_label(label)
    return dataset