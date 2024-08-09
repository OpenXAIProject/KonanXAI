
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

import sys


# darknet test 여기부터 시작
# from KonanXAI.models.model_import import model_import 
# from KonanXAI._core import darknet

# 리눅스 경로
# cfg_path = '/mnt/d/KonanXAI_implement_darknet/KonanXAI/yolov4-tiny.cfg'
# weight_path = '/mnt/d/KonanXAI_implement_darknet/KonanXAI/yolov4-tiny.weights'

# # 윈도우 darknet test
# cfg_path = 'D:\KonanXAI_implement_darknet\KonanXAI\yolov4-tiny.cfg'
# weight_path = 'D:\KonanXAI_implement_darknet\KonanXAI\yolov4-tiny.weights'
# image_path = 'D:\KonanXAI_implement_darknet\KonanXAI\dog.jpg'

# model = darknet.Network()
# model.load_model_custom(cfg_path, weight_path)


# image = darknet.open_image(image_path, (416, 416))
# model.forward_image(image)
# #print(len(model.layers[-1].get_output()))
# print(len(model.layers[-1].get_bboxes()))

from project.config import Configuration

config_path = 'D:\\KonanXAI_implement_darknet\\KonanXAI\\project\\config_darknet.yaml'
config = Configuration(config_path)