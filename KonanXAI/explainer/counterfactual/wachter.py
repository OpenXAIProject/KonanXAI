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

from KonanXAI.explainer.counterfactual import Counterfactual


from PIL import Image
import matplotlib.pyplot as plt

__all__ = ["Wachter"]
# ABCMeta 상속으로 해야하나?
class Wachter(Counterfactual):
    ''' explain something...
    
    '''
    def __init__(self, framework, model, dataset, config):
        Counterfactual.__init__(self, framework, model, dataset, config)


    def _perturb_input(self):
        self.cf_image = self.input

    def _define_loss_function(self):
        self.pdistance = nn.PairwiseDistance(p=0.2, eps = 1e-6, keepdim=False)
        self.criterion = nn.CrossEntropyLoss()
            

    def _define_optimizer(self):
        self.optimizer = optim.SGD(self.cf_image, lr = self.learning_rate, momentum = 0.9)



    def calculate(self):
        for iter in self.epoch:
            self._perturb_input()
            pred = self.model(self.cf_image)
            _, pred_label = torch.max(pred.data, 1)
            target_label = [self.target_label]
            target_label = torch.tensor(target_label, device = 'cuda:0')

            loss = self.coeff * self.criterion(pred, target_label) + self.pdistance(self.cf_image, self.input).sum()

            loss.backward()
            self.optimizer.step()
            cf_label = pred_label
            if (iter %100 == 0):
                print(f'loss {loss} target_label {target_label} predicted {pred_label}')
            
                save_image(self.cf_image, f'./cf_image_{iter}_{pred_label}.jpg')



    



