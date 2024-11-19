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
__all__ = ["GradientxInput"]
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

    def calculate(self,inputs=None,targets=None):
        if inputs != None:
            self.input = inputs
        if targets != None:
            self.label_index = targets
            
        self.get_saliency()
        if self.framework == 'torch':
            if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                for i, heatmap in enumerate(self.heatmaps):
                    self.heatmaps[i] = heatmap * self.input

                return self.heatmaps, self.bboxes
            else:
                self.heatmaps = self.heatmaps * self.input
                return self.heatmaps
        elif self.fraemwork == 'darknet':
            pass



                        

    