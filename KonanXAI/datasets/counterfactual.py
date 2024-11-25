from KonanXAI.datasets import Datasets
import glob
import os
import numpy as np
import cv2
__all__= ["Counterfactual"]

class Counterfactual(Datasets):
    def __init__(self, framework, src_path):
        super().__init__(framework = framework, src_path= src_path)
        self.dataset_name = 'counterfactual'
        self.framework = framework
        self.src_path = src_path
        
        
        
        
    def load_src_path(self):
        if self.mode == 0:
            train_path = glob(self.src_path+"/training/{self.label}/*.*")
            self.train_items = []
        
            for path in train_path:
                label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
                label = int(label)
                self.train_items.append((path, label))
        elif self.mode == 1:
            self.test_items = []
            test_path = glob(self.src_path+"/testing/{self.label}/*.*")# 수정 필요
        
            for path in test_path:
                label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
                label = int(label)
                self.test_items.append((path, label))