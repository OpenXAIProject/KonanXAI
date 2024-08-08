from .datasets import Datasets
import os
from glob import glob
class CUSTOM(Datasets):
    def __init__(self, src_path):
        super().__init__(src_path)
        self.src_path = src_path
        self.classes = 5
        self.make_cache = {}
        
    def load_src_path(self):
        train_path = glob(self.src_path+"/*.jpg")
        test_path = glob(self.src_path+"/*.jpg")# 수정 필요
        self.train_items = []
        self.test_items = []
        for path in train_path:
            # label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            label = -1
            self.train_items.append((path, label))

        for path in test_path:
            # label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            label = -1
            self.test_items.append((path, label))
            
    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, idx):
    #     image = darknet.open_image(self.data[idx], (640, 640))#(640, 640))
    #     return image