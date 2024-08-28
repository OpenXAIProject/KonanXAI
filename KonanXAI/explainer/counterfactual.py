import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable

from typing import Literal, List, Optional, Callable, Union
from 

import os
import json
import numpy as np
import cv2

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision import datasets


from PIL import Image
import matplotlib.pyplot as plt

# ABCMeta 상속으로 해야하나?
class Counterfactual:
    ''' explain something...
    
    '''
    def __init__(self, framework, model, origin_data, target_data, config):
        pass

    def 


