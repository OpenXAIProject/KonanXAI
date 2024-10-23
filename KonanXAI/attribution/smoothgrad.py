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


# Attribution 상속 지음
# yolo target_layer = [model, '23', 'cv1','conv']
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
        self.inputs = samples + self.noise_level * noise_sampling
        self.inputs = samples + self.noise_level * noise_sampling
        
    def calculate(self):
        self._gaussian_noise_sample()
        total_heatmap = []
        total_bboxes = []
        for i in range(self.sample_size):
            self.input = self.inputs[i].unsqueeze(0)
            self.get_saliency()
            if self.framework == 'torch':
                if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                    total_heatmap.append(self.heatmaps)
                    total_bboxes.append(self.bboxes)
                else:
                    total_heatmap.append(self.heatmaps)


        if self.framework == 'torch':
            if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                heatmaps = []
                bboxes = []
                num_box = min(len(total_heatmap[i]) for i in range(self.sample_size))
                for i in range(num_box):
                    bbox = torch.cat([torch.tensor(total_bboxes[j][i]).unsqueeze(0) for j in range(self.sample_size)], dim=0)
                    bbox = torch.mean(bbox, dim=0)
                    bboxes.append(bbox)
                    heatmap = torch.cat([total_heatmap[j][i] for j in range(self.sample_size)], dim=0)
                    heatmap = torch.mean(heatmap, dim = 0).unsqueeze(0)
                    heatmaps.append(heatmap)
                    
                return heatmaps, bboxes
            else:
                total_heatmap = torch.cat(total_heatmap, dim=0)
                total_heatmap = torch.mean(total_heatmap, dim=0).unsqueeze(0)
                return total_heatmap
        elif self.fraemwork == 'darknet':
            pass

    def _yolo_get_bbox_pytorch(self):
        self.input.requires_grad=True
        for param in self.model.parameters():
            param.requires_grad = True

        self.preds_origin, raw_logit = self.model(self.input)
        self.logits_origin = torch.concat([data.view(-1,self.preds_origin.shape[-1])[...,5:] for data in raw_logit],dim=0)
        with torch.no_grad():
            self.preds, logits, self.select_layers = non_max_suppression(self.preds_origin, self.logits_origin.unsqueeze(0), conf_thres=0.25, model_name = self.model_name)
        self.index_tmep = yolo_choice_layer(raw_logit, self.select_layers)

    def _yolo_backward_pytorch(self):
        self.bboxes = []
        self.heatmaps = []
        for cls, sel_layer, sel_layer_index in zip(self.preds[0], self.select_layers, self.index_tmep):
            self.model.zero_grad()
            if self.input.grad != None:
                self.input.grad.zero_()
            self.logits_origin[sel_layer][int(cls[5].item())].backward(retain_graph=True)

            heatmap = self.input.grad.clone().detach()
            self.heatmaps.append(heatmap)
            self.bboxes.append(cls[...,:4].detach().cpu().numpy())
            