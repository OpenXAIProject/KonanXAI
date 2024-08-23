import darknet  
# from ..lib.core import dtrain
# from ..lib.core import pytorch
import random
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms



class Datasets:
    def __init__(self, framework, src_path, label_path=None):
        self.framework = framework
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
        
        if idx >= len(self):
            raise IndexError
        data = self.generator.send(idx)
        next(self.generator)
            
        return data
    
    # resize
    def set_fit_size(self, size=(224, 224)):
        self.fit = size
        
    def get_custom(self, idx):
        return None
    
    def gen(self):
        while True:
            idx = yield
            if self.framework == 'darknet':
                origin_img = cv2.imread(self.train_items[idx][0])
                origin_img = cv2.resize(origin_img,self.fit)
                data = darknet.open_image(self.train_items[idx][0], self.fit)
                data.origin_img = origin_img
                data.im_size = self.fit
                yield data
            else:
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
                            data = Image.open(xp)
                            # data = cv2.imread(xp,cv2.COLOR_BGR2RGB)
                        if self.fit is not None:
                            compose_resize = transforms.Compose([
                                transforms.Resize(self.fit),
                                transforms.ToTensor()
                            ])
                        #     # data = cv2.resize(data, self.fit, interpolation=cv2.INTER_CUBIC)
                        # # data = np.transpose(data, (2, 0, 1))
                        # # x = torch.tensor(data, dtype=torch.float32) / 255.
                        x = compose_resize(data)
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
                yield (xtensor, ytensor, custom, self.fit)
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
        self.mode = 1
    # toPytorch
    def toTensor(self):
        self.data_type = 0

    # toDarknet
    def toDarknet(self):
        self.data_type = 1

    # toDTrain
    def toDtrain(self):
        pass