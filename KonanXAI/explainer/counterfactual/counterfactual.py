import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable

from typing import Literal, List, Optional, Callable, Union

import os
import json
import numpy as np
import cv2

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image


from PIL import Image
import matplotlib.pyplot as plt

# ABCMeta 상속으로 해야하나?
class Counterfactual:
    ''' explain something...
    
    '''
    def __init__(self, framework, model, dataset, config):
        self.algorithm = config['algorithm']
        self.input_index = config['input_index']
        self.input = dataset[self.input_index][0]
        self.target_label = config['target_label']
        self._lambda = config['lambda']
        self.epoch = config['epoch']
        self.learning_rate = config['learning_rate']
        

    def _perturb_input(self):
        self.cf_image = self.input


    def _define_loss_function(self):
        pass
            

    def _define_optimizer(self):
        pass 

    def calculate(self):
        pass



    



