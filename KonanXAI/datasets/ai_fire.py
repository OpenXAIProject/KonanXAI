from KonanXAI.datasets import Datasets
import glob
import os
import numpy as np
import cv2
class AI_FIRE(Datasets):
    def __init__(self, framework, src_path):
        super().__init__(framework = framework, src_path= src_path)
        self.classes = 5
        self.dataset_name = 'aifire'
        self.framework = framework
        self.src_path = src_path
        self.mask_cache = {}
        
    def get_custom(self, idx):
        if self.mode == 0:
            s = idx * self.batch
            path = self.train_items[s: s+self.batch]
            mask = []
            for (xp, _) in path:
                if xp not in self.mask_cache:
                    mask_path = os.path.dirname(os.path.dirname(xp))
                    mask_path = xp.replace(mask_path, mask_path + "_mask")
                    data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if self.fit is not None:
                        data = cv2.resize(data, self.fit, interpolation=cv2.INTER_CUBIC)
                    self.mask_cache[xp] = data
                data = self.mask_cache[xp]
                mask.append(data)
            mask = np.stack(mask)
            # mask = np.where(mask > 0, 1, 0)      
            return mask
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
        test_path = load_image(self.src_path.replace("train","test_lite"))
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