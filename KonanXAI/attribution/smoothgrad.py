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
from KonanXAI.utils.sampling import gaussian_noise
__all__ = ["SmoothGrad"]
class SmoothGrad(Gradient):
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
        self.std = config['std']
        self.noise_level = config['noise_level']
        self.sample_size = config['sample_size']

    def _gaussian_noise_sample(self):
        self.std = self.std * (torch.max(self.input) - torch.min(self.input))
        noise_sampling = gaussian_noise(self.input.shape, self.std, self.sample_size).to(self.device)
        samples = torch.repeat_interleave(self.input, self.sample_size, dim = 0)
        self.input = samples + self.noise_level * noise_sampling
        
    def calculate(self):
        self._gaussian_noise_sample()
        self.get_saliency()
        self.saliency = (torch.sum(self.saliency, dim=0)/self.sample_size).unsqueeze(0)
        return self.saliency