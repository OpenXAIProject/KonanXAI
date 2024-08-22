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
import h5py

# ABCMeta 상속으로 해야하나?
class Clustering:
    ''' explain something...
    
    '''
    def __init__(self, 
                 framework,
                 model,
                 dataset,
                 algorithm):
        self.framework = framework
        self.model = model
        self.dataset = dataset
        self.algorithm = algorithm
        self.len_dataset = len(self.dataset)
        if self.framework == 'darknet':
            self.input_size = self.dataset[0].shape
        else:
            self.input_size = self.dataset[0][0].shape[2:4]
        self.dataset_file_path = '.../clustering_result/dataset.h5'

    def create_dataset_h5_file(self):
        self.dataset_file = h5py.File(self.dataset_file_path, 'w')
        self.dataset_file.create_dataset(
            'data',
            shape = (self.len_dataset,) + tuple(self.input_size),
            dtype = 'float32'
        )
        self.dataset_file.create_dataset(
            'label',
            shape = (self.len_dataset,),
            dtype = 'uint16'
        )
        return self.dataset_file
    
    def append_dataset(self, index, batch, sample, label):
        self.dataset_file['data'][index:batch+index] = sample
        self.dataset_file['label'][index:batch+index] = label

    def apply(self):
        with self.create_dataset_h5_file(self.)

