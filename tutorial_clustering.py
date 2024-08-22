
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
from glob import glob

import PIL.Image
from PIL import Image

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision import datasets

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#from KonanXAI.explainer.adversarial import FGSM
from KonanXAI.models import model_info

# 아래 호출 경로 이름을 바꾸던가 좀 수정해야...
import KonanXAI
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset
import project

import darknet

from KonanXAI.attribution import GradCAM
import h5py

# 윈도우에서 확인 darknet.so 파일이 없어서..
# framework = 'torch'
# source = 'github'
# repo_or_dir = 'ultralytics/ultralytics'
# model_name = 'yolov5s'
# cache_or_local = 'cache'
# weight_path = 'D:\\KonanXAI_implement_darknet\\yolov5s.pt'
# cfg_path = None
# target_layer = {'0':['model','24','m','0'],'1':['model','24','m','1'],'2':['model','24','m','2']}


framework = 'darknet'
source = 'local'
repo_or_dir = 'D:\\KonanXAI_implement_darknet\\xai_darknet'
model_name = 'yolov4'
cache_or_local = None
weight_path = 'D:\\3852.weights'
# custom 모델 cfg 불러오는 경우
cfg_path = 'D:\\3852.cfg'
target_layer = None

model = model_import(framework = framework, 
                     source = source, 
                     repo_or_dir = repo_or_dir, 
                     model_name = model_name, 
                     cache_or_local = cache_or_local,
                     weight_path = weight_path,
                     cfg_path=cfg_path)

print(model)
#print(model.model_name)
#print(type(model))
#리눅스 data_path
#data_path = "/mnt/d/dataset/military_data/military_data/107mm/"
#윈도우 data_path
data_path = "D:\\dataset\\military_data\\military_data\\107mm\\"
# data_path2 = "D:\\KonanXAI_implement_darknet\\yolov5\\data\\images\\"
# data_loader = load_dataset(framework, data_path = data_path, 
#                               data_type = 'CUSTOM', resize = (640, 640))

# img_save_path = 'D:\\KonanXAI_implement_darknet\\KonanXAI\\result\\test.jpg'
# print(target_layer['0'])
# # 데이터 
# # input : (batch, channel, height, width) torch.Tensor
# gradcam = GradCAM(framework, model, data_loader[0][0], target_layer)
# heatmap, bboxes = gradcam.calculate()
# heatmap = gradcam.get_heatmap(img_save_path)
# print(len(bboxes))




# darknet gradcam test
#darknet_image = darknet.open_image(data_path2, (416, 416))
data_loader = load_dataset(framework, data_path = data_path, 
                              data_type = 'CUSTOM', resize = (640, 640))

for i, data in enumerate(data_loader):
    print(data_loader.train_items[1])

# image_path = 'D:\\KonanXAI_implement_darknet\\KonanXAI\dog.jpg'
# image = darknet.open_image(image_path, (416,416))
# model = darknet.Network()
# model.load_model_custom(cfg_path, weight_path)
# model.forward_image(image)
# print(model.layers[-1].get_bboxes())
# target_layer 마지막 레이어로 정해져 있는거 같은데?
#target_layer =None


# gradcams = []
# for i, data in enumerate(data_loader):
#     print(1, data_loader.train_items[i][0])
#     img_path = data_loader.train_items[i][0].split('\\')
#     root = 'D:\\gradcam_result\\' + img_path[-2]
#     if os.path.isdir(root) == False:
#         print(root)
#         os.mkdir(root)
    

#     img_save_path = root + '\\' + img_path[-1]
#     gradcam = GradCAM(framework, model, data_loader[0], target_layer)
#     gradcam.calculate()
#     gradcams.append(gradcam)
#     gradcam.get_heatmap(img_save_path)








