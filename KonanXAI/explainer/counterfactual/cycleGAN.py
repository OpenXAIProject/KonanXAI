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

__all__ = ["CycleGAN_CF"]


# ABCMeta 상속으로 해야하나?
class CycleGAN_CF(Counterfactual):
    ''' explain something...
    
    '''
    def __init__(self, framework, model, dataset, config):
        Counterfactual.__init__(self, framework, model, dataset, config)
        
        self.gen_AtoB_weight_path = config['gen_AtoB_weight_path']
        self.gen_BtoA_weight_path = config['gen_BtoA_weight_path']
        self.disc_A_weight_path = config['disc_A_weight_path']
        self.disc_B_weight_path = config['disc_B_weight_path']
        self.CF_gen_AtoB_weight_path = config['CF_gen_AtoB_weight_path']
        self.CF_gen_BtoA_weight_path = config['CF_gen_BtoA_weight_path']
        self.CF_disc_A_weight_path = config['CF_disc_A_weight_path']
        self.CF_disc_B_weight_path = config['CF_disc_B_weight_path']
        self.cycleGAN_lr = config['cycleGAN_learning_rate']
        self.cycleGAN_epochs = config['cycleGAN_epochs']
        self.CF_lr = config['CF_learning_rate']
        self.CF_epochs = config['CF_epochs']
        self._lambda = config['lambda'] 
        self.mu = config['mu']
        self.gamma = config['gamma']
        
    def _make_cycleGAN_dataset(self):

        

    def _perturb_input(self):
        self.cf_image = self.input

    def _CycleGAN(self):


    def _define_loss_function(self):
        pass
            

    def _define_optimizer(self):
        pass 

    def apply(self):
        self.cycleGAN = _CycleGAN
        pass


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

    



