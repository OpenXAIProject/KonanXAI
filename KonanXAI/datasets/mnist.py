from KonanXAI.datasets import Datasets
from KonanXAI.utils.data_convert import convert_tensor
try:
    import darknet  
except ImportError as e:
    print(f"Darknet 관련 Dataset 사용 불가.{e}")
import os
import cv2
from glob import glob
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
__all__= ["MNIST"]

class MNIST(Datasets):
    def __init__(self, framework, src_path, label = None):
        super().__init__(framework, src_path, label = None)
        self.framework = framework
        self.src_path = src_path
        self.classes = 10
        self.dataset_name = 'mnist'
        self.make_cache = {}
        
    def load_src_path(self):
        train_path = glob(self.src_path+"/training/*/*.*")
        test_path = glob(self.src_path+"/testing/*/*.*")# 수정 필요
        self.train_items = []
        self.test_items = []
        for path in train_path:
            label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            label = int(label)
            self.train_items.append((path, label))

        for path in test_path:
            label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            label = int(label)
            self.test_items.append((path, label))

    