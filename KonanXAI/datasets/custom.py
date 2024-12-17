from KonanXAI.datasets import Datasets
import os
from glob import glob
from pathlib import Path
__all__= ["CUSTOM"]
class CUSTOM(Datasets):
    def __init__(self, framework, src_path, label = None):
        super().__init__(framework, src_path, label = None)
        self.framework = framework
        self.src_path = src_path
        self.dataset_name = 'imagenet'
        self.classes = 1000
        self.make_cache = {}
        
    def load_src_path(self):
        def load_image(src_path):
            extension_types = ['*.jpg','*.png','*.jpeg', '*.JPEG']
            file_list = []
            for types in extension_types:
                file_list.extend(glob(os.path.join(src_path, types)))
            return [str(file) for file in file_list]
       
        train_path = load_image(self.src_path)
        test_path = load_image(self.src_path)# 수정 필요
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