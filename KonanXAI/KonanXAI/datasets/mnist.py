from KonanXAI.datasets import Datasets
import os
from glob import glob

class MNIST(Datasets):
    def __init__(self, src_path):
        super().__init__(src_path)
        self.src_path = src_path
        self.classes = 10
        self.make_cache = {}
        
    def load_src_path(self):
        train_path = glob(self.src_path+"/training/*/*.*")
        test_path = glob(self.src_path+"/testing/*/*.*")# 수정 필요
        self.train_items = []
        self.test_items = []
        for path in train_path:
            label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            label = int(label) - 1
            self.train_items.append((path, label))

        for path in test_path:
            label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            label = int(label) - 1
            self.test_items.append((path, label))