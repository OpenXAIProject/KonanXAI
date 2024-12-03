from KonanXAI.datasets import Datasets
import glob
import os
import numpy as np
import cv2
__all__= ["CFdatasets"]


class CFDatasets(Datasets):
    def __init__(self, framework, src_path):
        super().__init__(framework, src_path)
        self.classes = 1
        self.dataset_name = 'counterfactual'
        self.framework = framework
        self.src_path = src_path
        self.labels = 0
        
        
    def set_label(self, label):
        self.labels = label
        
        
        
    def load_src_path(self):
        train_path = glob(self.src_path+"/training/{self.label}/*.*")
        test_path = glob(self.src_path+"/testing/{self.label}/*.*")# 수정 필요
        self.train_items = []
        self.test_items = []
        for path in train_path:
            self.train_items.append((path, self.label))

        for path in test_path:
            self.test_items.append((path, self.label))