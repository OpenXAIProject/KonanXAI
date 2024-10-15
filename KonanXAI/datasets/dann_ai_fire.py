from KonanXAI.datasets import Datasets
import glob
import os
import numpy as np
import cv2
import torch
class DANN_AI_FIRE(Datasets):
    def __init__(self, framework, src_path):
        self.real_cache = {}
        super().__init__(framework = framework, src_path= src_path)
        self.framework = framework
        self.src_path = src_path
        self.dataset_name = 'aifire'
        self.classes = 5
    
    def get_custom(self, idx):
        if self.mode == 0:
            s = idx * self.batch
            path = self.real_items[s: s+self.batch]
            real_batch = []
            for (xp, _) in path:
                if xp not in self.real_cache:
                    real_path = xp
                    data = cv2.imread(real_path)
                    if self.fit is not None:
                        data = cv2.resize(data, self.fit, interpolation=cv2.INTER_CUBIC)
                    data = np.transpose(data,(2,0,1))
                    x = torch.tensor(data, dtype=torch.float32) / 255.
                    self.real_cache[xp] = x
                x = self.real_cache[xp]
                real_batch.append(x)
            real_tensor = torch.stack(real_batch)
            # mask = np.where(mask > 0, 1, 0)      
            return real_tensor
        else:
            return None
        
    def load_src_path(self):
        def load_image(src_path):
            extension_types = ['*.jpg','*.png','*.jpeg']
            file_list = []
            for types in extension_types:
                file_list.extend(glob.iglob(src_path+"/**/"+types, recursive=True))
            return [str(file) for file in file_list]
       
        train_path = load_image(self.src_path)
        real_path = load_image(self.src_path.replace("train","real"))
        test_path = load_image(self.src_path.replace("train","test_lite"))
        self.train_items = []
        self.real_items = []
        self.test_items = []
        for path in train_path:
            label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            label = int(label) - 1
            self.train_items.append((path, label))

        for path in real_path:
            label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            label = int(label) - 1
            self.real_items.append((path, label))
        
        for path in test_path:
            label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            label = int(label) - 1
            self.test_items.append((path, label))