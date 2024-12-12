from KonanXAI.datasets import Datasets
from glob import glob
import os
import numpy as np
import cv2
__all__= ["CFdatasets"]


class CFDatasets(Datasets):
    def __init__(self, framework, src_path, label = None):
        Datasets.__init__(self, framework, src_path, label = None)
        self.classes = 1
        self.dataset_name = 'counterfactual'
        self.make_cache = {}
        self.label = label
        self.load_src_path()

   

    def load_src_path(self):
        train_path = glob(self.src_path+"/training/" + f"{self.label}" + "/*.*")
        test_path = glob(self.src_path+"/testing/" + f"{self.label}" + "/*.*")# 수정 필요
        self.train_items = []
        self.test_items = []
        for path in train_path:
            self.train_items.append((path, self.label))

        for path in test_path:
            self.test_items.append((path, self.label))