from ..lib.core import darknet
# from ..lib.core import dtrain
# from ..lib.core import pytorch
import random
import cv2
import numpy as np
import torch
class Datasets:
    def __init__(self, src_path, label_path=None):
        if label_path is None:
            self.label_path = src_path
        self.data_type = 0
        self.src_path = src_path
        self.train_items = None
        self.test_items = None
        self.cache = {}
        self.batch = 1
        self.classes = 1
        self.fit = None
        self.mode = 0
        self.generator = self.gen()
        next(self.generator)
        self.load_src_path()
        
    def load_src_path(self):
        pass
    
    def __len__(self):
        if self.mode == 0:
            return len(self.train_items) // self.batch
        else:
            return len(self.test_items) // self.batch

    def __getitem__(self, idx):
        if self.data_type == 0:
            if idx >= len(self):
                raise IndexError
            data = self.generator.send(idx)
            next(self.generator)
            
        elif self.data_type == 1:
            data = darknet.open_image(self.train_items[idx], self.fit)#(640, 640))
        return data
    
    # resize
    def set_fit_size(self, size=(224, 224)):
        self.fit = size
        
    def get_custom(self, idx):
        return None
    
    def gen(self):
        while True:
            idx = yield
            s = idx * self.batch
            if self.mode == 0:
                path = self.train_items[s : s+self.batch]
            else:
                path = self.test_items[s: s+self.batch]
            xbatch = []
            ybatch = []
            for (xp, yp) in path:
                data = None
                if isinstance(xp,tuple):
                    data = xp[1]
                    xp = xp[0]
                if xp not in self.cache: 
                    if isinstance(data,np.ndarray):
                        data = data
                    else:
                        data = cv2.imread(xp)
                    if self.fit is not None:
                        data = cv2.resize(data, self.fit, interpolation=cv2.INTER_CUBIC)
                    data = np.transpose(data, (2, 0, 1))
                    x = torch.tensor(data, dtype=torch.float32) / 255.
                    if isinstance(yp, list):
                        y = torch.tensor([yp], dtype=torch.float32)
                    else:
                        y = torch.tensor([yp], dtype=torch.long)
                    # y[yp] = 1.0
                    self.cache[xp] = (x, y)
                # Normalize
                x, y = self.cache[xp]
                # Label
                xbatch.append(x)
                ybatch.append(y)
            xtensor = torch.stack(xbatch)
            ytensor = torch.stack(ybatch).squeeze()
            custom = self.get_custom(idx)
            yield (xtensor, ytensor, custom)
    # shuffle
    def shuffle(self):
        if self.mode == 0:
            random.shuffle(self.train_items)
        else:
            random.shuffle(self.test_items)
    def set_batch(self,size:int):
        self.batch = size
    def set_train(self):
        self.mode = 0
    def set_test(self):
        self.model = 1
    # toPytorch
    def toTensor(self):
        self.data_type = 0

    # toDarknet
    def toDarknet(self):
        self.data_type = 1

    # toDTrain
    def toDtrain(self):
        pass