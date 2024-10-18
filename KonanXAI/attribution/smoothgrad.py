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
        
    def calculate(self, inputs=None, targets=None):
        if inputs != None:
            self.input = inputs
        if targets != None:
            self.label_index = targets
        self._gaussian_noise_sample()
        self.get_saliency()
        if self.framework == 'torch':
            if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                for i, heatmap in enumerate(self.heatmaps):
                    self.heatmaps[i] = torch.mean(heatmap, dim=0).unsqueeze(0)
                return self.heatmaps, self.bboxes
            else:
                self.heatmaps = torch.mean(self.heatmaps, dim=0).unsqueeze(0)
                return self.heatmaps
        elif self.fraemwork == 'darknet':
            pass

    def _yolo_get_bbox_pytorch(self):
        self.input.requires_grad=True
        for param in self.model.parameters():
            param.requires_grad = True

        self.preds_origin, raw_logit = self.model(self.input)
        self.logits_origin = []
        self.preds = []
        self.select_layers = []
        self.index_tmep = []
        for i in range(self.preds_origin.shape[0]):
            logits_origin = torch.concat([data.view(-1,self.preds_origin[i].shape[-1])[...,5:] for data in raw_logit[0][i]],dim=0)
            self.logits_origin.append(logits_origin)
        #self.logits_origin = torch.concat([data.view(-1,self.preds_origin.shape[-1])[...,5:] for data in raw_logit],dim=0)
        
            with torch.no_grad():
                preds, logits, select_layers = non_max_suppression(self.preds_origin[i], self.logits_origin[0][i], conf_thres=0.25, model_name = self.model_name)
                self.preds.append(preds)
                self.select_layers.append(select_layers)
                index_tmep = yolo_choice_layer(raw_logit[i], self.select_layers[i])
                self.index_tmep.append(index_tmep)

    def _yolo_backward_pytorch(self):
        self.bboxes = []
        self.heatmaps = []
        for cls, sel_layer, sel_layer_index in zip(self.preds[0], self.select_layers, self.index_tmep):
            self.model.zero_grad()
            self.logits_origin[sel_layer][int(cls[5].item())].backward(retain_graph=True)

            heatmap = self.input.grad
            self.heatmaps.append(heatmap)
            self.bboxes.append(cls[...,:4].detach().cpu().numpy())
