from ..lib.core import darknet
# from ..lib.core import dtrain
# from ..lib.core import pytorch

import cv2
import numpy as np
class Datasets():
    def __init__(self):
        self.path = ""

    def __getitem__(self, idx):
        return
    
    # resize
    def resized(self):
        pass

    # shuffle
    def shuffle(self):
        pass
    
    # toPytorch
    def toTensor(self):
        pass

    # toDarknet
    def toDarknet(self):
        pass

    # toDTrain
    def toDtrain(self):
        pass