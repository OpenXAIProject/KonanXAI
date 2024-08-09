
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
from KonanXAI.datasets import MNIST
import project


mnist = globals().get('MNIST')
print(mnist)

# # 1) github repository download 예제
# # 일단 url 로 모델 다운로드에서 막힌 상태
# #task = 'image'  # not yet
# # device 설정 넣어야하나?
# # device 설정을 넣으려면 method로 짜긴 해야겠네...
# framework = 'torch'
# source = 'github'
# repo_or_dir = 'ultralytics/yolov5/'
# model_name = 'yolov5s'
# cache_or_local = '/mnt/d/KonanXAI_implement_example2/'
# weight_path = '/mnt/d/KonanXAI_implement_example2'

# 2) source가 local, ultralytics/yolov5로 테스트 -> 잘됨. 
# 너무 쉽게 구현된 거 아닌가..
# framework = 'torch'
# source = 'local'
# repo_or_dir = '/mnt/d/KonanXAI_implement_yolov5/yolov5/'
# model_name = 'yolov5s'
# cache_or_local = None
# weight_path = '/mnt/d/KonanXAI_implement_yolov5/yolov5s.pt'

# 3) source: local, ultralytics/ultralytics 테스트 -> 일단 됨
# YOLO 클래스 바로 import 해오는 방법으로 잘되는데
# hubconf.py 모델에서 작성하는 걸로 다시 수정해야?
# framework = 'torch'
# source = 'local'
# repo_or_dir = '/mnt/d/KonanXAI_implement_example2/ultralytics/'
# model_name = 'yolov8s'
# cache_or_local = None
# weight_path = '/mnt/d/KonanXAI_implement_example2/yolov8s.pt'

# cache_or_local 이라고 명칭을 바꿀까? 

# 4) source:torchvision, repo_or_dir = None, model_name: EfficientNet-bo -> 잘됨
# framework = 'torch'
# source = 'torchvision'
# repo_or_dir = None
# model_name = 'EfficientNet_b0'
# cache_or_local = None
# weight_path = None   # weight_path = None 이면 pretrained=True 자동으로 들어가게 일단 해놓을까?

# # 5) framework: darknet, source:github, model_name: yolov4-tiny
# # 리눅스 cmake 안해서 윈도우에서 테스트중
# # cache 저장 경우도 만들어야
# framework = 'darknet'
# source = 'github'
# repo_or_dir = 'AlexeyAB/darknet'
# model_name = 'yolov4-tiny'
# cache_or_local = 'D:\\KonanXAI_implement_darknet\\'
# weight_path = 'D:\\weights\yolov4-tiny.weights'
# # custom 모델 cfg 불러오는 경우
# cfg_path = 'D:\\KonanXAI_implement_darknet\\KonanXAI\yolov4-tiny.cfg'


# 6) framework: darknet, source:local, model_name: yolov4-tiny
# 리눅스 cmake 안해서 윈도우에서 테스트중
# cache 저장 경우도 만들어야?
framework = 'darknet'
source = 'local'
repo_or_dir = 'D:\\KonanXAI_implement_darknet\\xai_darknet'
model_name = 'yolov4-tiny'
cache_or_local = 'D:\\KonanXAI_implement_darknet\\'
weight_path = 'D:\\weights\yolov4-tiny.weights'
# custom 모델 cfg 불러오는 경우
cfg_path = 'D:\\KonanXAI_implement_darknet\\KonanXAI\yolov4-tiny.cfg'


model = model_import(framework = framework, 
                     source = source, 
                     repo_or_dir = repo_or_dir, 
                     model_name = model_name, 
                     cache_or_local = cache_or_local,
                     weight_path = weight_path,
                     cfg_path = cfg_path)

print(model)

## darknet에서 쓰는 경우
## dataset load
data_path = "D:\\dataset\\military_data\\military_data\\107mm\\"
label_path = "D:\\dataset\\military_data\\military_raw_data"


darknet_loader = load_dataset(framework, data_path = data_path, 
                              data_type = 'CUSTOM', resize = (416, 416))
print(darknet_loader)



# 기존 torch 모델 데이터로더 구성 예시
# dataset = datasets.ImageForder(data_path, transform = transform)
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# train_loader = torch.utils.data.Dataloader(train_dataset, batch_size = 16, shuffle = True)




# 이렇게 구현할까, 말까?
# label_path = Datasets(data_path = data_path, label_path = label_path,
#                       train_or_test = 'test', batch = 1, resize = (416,416),
#                       transform = None, data_type = 'CUSTOM', framework = 'darknet') 
# #dataset = CUSTOM(data_path)



# or model = models.load_model.VGG16(...) 로 사용가능


# image = torch.rand([1,3,224,224])
# prediction = model(image)
# print(prediction)



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