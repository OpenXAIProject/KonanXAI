import torch
import numpy as np
import cv2

# weight_path = "./resnet50_military_data_epoch10_cuda.pt"

# model = torch.hub.load('pytorch/vision:v0.11.0', 'resnet50')
# weight = torch.load(weight_path)
# key = list(weight.keys())[-1]
# print(key)
# print(weight[key].shape[0])

import torchvision
import torch.nn as nn
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image 
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.cluster import KMeans


framework = 'torch'
source = 'torchvision'
repo_or_dir = None
data_path = "../dataset/MNIST"
data_type = 'MNIST'
model_name = 'resnet50'
data_resize = [224,224]
cache_or_local = None
weight_path = None   # weight_path = None 이면 pretrained=True 자동으로 들어가게 일단 해놓을까?
cfg_path = None

device = torch.device('cuda:0')
# model = model_import(framework, source, repo_or_dir,
#                                   model_name, cache_or_local, 
#                                   weight_path)
dataset = load_dataset(framework, data_path = data_path,
                                    data_type = data_type, resize = data_resize)

for i in enumerate(dataset):
    dataset.train_items[i][1]

print(dataset[0])