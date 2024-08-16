
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
import sys

sys.path.append('/mnt/d/KonanXAI_implement_example2/ultralytics/')

# from ultralytics.models.yolo.model import YOLO

# model = YOLO()
# print(model)


import sys, os
import yaml
import pathlib
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from typing import Callable

import os
import json
import numpy as np
import cv2

import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision import datasets

#from PIL import Image
import matplotlib.pyplot as plt

from collections import defaultdict

from inspect import signature

#from config.config_parser import Configuration

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from torchvision.transforms import Normalize

import enum
# Yolo 모델에 필요한 import #
from yolov5.models.yolo import Detect, Model

from yolov5.models.common import C3, Conv, Bottleneck, Concat, SPPF
from yolov5.models.experimental import Ensemble

from yolov5.models.yolo import Detect, Model

from yolov5.models.experimental import Ensemble
#######################

class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        
        y = torch.cat(y,1)
        return y, None
