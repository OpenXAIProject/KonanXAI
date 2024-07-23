
# 예시 파일

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable



import os
import json
import numpy as np
import cv2

import PIL.Image
from PIL import Image

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision import datasets



import matplotlib.pyplot as plt

#from KonanXAI.explainer.adversarial import FGSM
from KonanXAI.models import model_info

# 아래 호출 경로 이름을 바꾸던가 좀 수정해야...
from KonanXAI.models.model_import import model_import 

framework = 'torch'
source = 'github'
repo_or_dir = 'ultralytics/ultralytics'
model_name = 'yolov8s'
local_path_to_load = '/mnt/d/KonanXAI_implement_example2/'
weight_path = '/mnt/d/KonanXAI_implement_example2'
model = model_import(framework = framework, 
                     source = source, 
                     repo_or_dir = repo_or_dir, 
                     model_name = model_name, 
                     local_path_to_load = local_path_to_load,
                     weight_path = weight_path)
# or model = models.load_model.VGG16(...) 로 사용가능
print(model)



# #data를 불러오긴 해야됨
# dataloader = ...
# loss_ft = torch.nn.CrossEntropyLoss()
# #adversarial_example = FGSM.generate_example(epsilon = 0.007, loss = loss_ft)


# # robust 모델 얻을 때



# # gradcam 적용 예시
# # '..' : model의 repository name 기준

# model = models.load_model.torch('ResNet')

# # yolo의 경우 torch.hub yolov5s import에 대응되게..?
# # model = models.load_model.darknet('yolov5s')