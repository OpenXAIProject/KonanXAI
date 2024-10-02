from KonanXAI._core.pytorch.yolov5s.utils import non_max_suppression, yolo_choice_layer
#from ..attribution import Attribution
from KonanXAI.utils import *
#from ....models import XAIModel
from KonanXAI.datasets import Datasets
from KonanXAI.attribution import Gradient
import darknet 
import torch
import numpy as np
import cv2
import torch.nn.functional as F

# Attribution 상속 지음
# yolo target_layer = [model, '23', 'cv1','conv']
class GradientxInput(Gradient):
    def __init__(
            self, 
            framework, 
            model, 
            input, 
            config):
        '''
        input: [batch, channel, height, width] torch.Tensor 
        '''
        Gradient.__init__(self, framework, model, input, config)


                        

    