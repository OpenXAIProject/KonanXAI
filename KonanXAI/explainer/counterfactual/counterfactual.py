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

__all__ = ["Counterfactual"]
# ABCMeta 상속으로 해야하나?
class Counterfactual:
    ''' explain something...
    
    '''
    def __init__(self, framework, model, dataset, config):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.framework = framework
        self.model = model
        self.model_name = self.model.model_name
        self.input_label = config['input_label']
        self.target_label = config['target_label']
        self.dataset = dataset
        

    def _perturb_input(self):
        self.cf_image = self.input


    def _define_loss_function(self):
        pass
            

    def _define_optimizer(self):
        pass 

    def calculate(self):
        pass



    



