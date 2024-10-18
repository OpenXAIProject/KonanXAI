from KonanXAI._core.pytorch.yolov5s.utils import non_max_suppression, yolo_choice_layer
#from ..attribution import Attribution
from KonanXAI.utils import *
#from ....models import XAIModel
from KonanXAI.datasets import Datasets
import darknet 
import torch
import numpy as np
import cv2
import torch.nn.functional as F

# Attribution 상속 지음
# yolo target_layer = [model, '23', 'cv1','conv']
class Gradient:
    def __init__(
            self, 
            framework, 
            model, 
            input, 
            config):
        '''
        input: [batch, channel, height, width] torch.Tensor 
        '''

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.framework = framework
        self.model = model
        self.model_name = self.model.model_name
        if framework.lower() == "darknet":
            self.input = input
            self.input_size = self.input.shape
        else:
            self.input = input[0].to(self.device)
            self.input_size = self.input.shape[2:4]

                        
    def get_saliency(self):
        if self.framework == 'torch':
            if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                self._yolo_get_bbox_pytorch()
                self._yolo_backward_pytorch()
                
                
            else:
                self.model.eval()
                self.input.requires_grad = True
                #self.input.retain_grad = True

                logits = self.model(self.input)
                target = torch.zeros_like(logits)
                for i in range(target.shape[0]):
                    target[i][torch.argmax(logits[i]).detach().cpu()] = 1
                self.model.zero_grad()
                logits.backward(target)
                self.heatmaps = self.input.grad
                #self.heatmaps = torch.sum(self.heatmaps, dim=1)

        elif self.framework == 'darknet':
            pass

    
    
    def calculate(self):
        self.get_saliency()
        if self.framework == 'torch':
            if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                return self.heatmaps, self.bboxes
            else:
                return self.heatmaps
        elif self.fraemwork == 'darknet':
            pass
    
    # Darknet
    def _yolo_get_bbox_darknet(self):
        self.bboxes = []
        self.bbox_layer = {}
        for i, layer in enumerate(self.model.layers):
            if layer.type == 28:
            # 아래 코드 에러
            #if layer.type == darknet.LAYER_TYPE.YOLO:
                # TODO - Threadhold 관련은 config 통합 후 진행, 현재는 정적

                boxes = layer.get_bboxes(threshold=0.5)
                for box in boxes:
                    self.bbox_layer[box.entry] = i
                    # print(f"where is box: {i}")
                # Concat
                
                self.bboxes += boxes
        # TODO - NMS, 여기도 Threshold 정적
        if len(self.bboxes) > 1:
            self.bboxes = darknet.non_maximum_suppression_bboxes(self.bboxes, iou_threshold=0.5)

    def _yolo_backward_darknet(self):
        for box in self.bboxes:
            i = self.bbox_layer[box.entry]
            # 여기서는 i-1을 쓰고 gradcampp 에서는 i를 쓰는 이유?
            target_layer = self.model.layers[i -1]
            out = target_layer.get_output()
            self.model.zero_grad()
            # feature index
            stride = target_layer.out_w * target_layer.out_h
            idx = box.entry + (5 + box.class_idx) * stride
            # set delta
            target_layer.delta[idx] = out[idx]
            self.model.backward()
            # Get Features
            # for target in target_layer:
    
            feature = torch.Tensor(target_layer.get_output())
            gradient = torch.Tensor(target_layer.get_delta())
            feature = feature.reshape((-1, target_layer.out_w, target_layer.out_h)).unsqueeze(0)
            gradient = gradient.reshape((-1, target_layer.out_w, target_layer.out_h)).unsqueeze(0)
            self.feature.append(feature)
            self.gradient.append(gradient)
        
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
            self.logits_origin[sel_layer][int(cls[5].item())].backward(retain_graph=True)

            heatmap = self.input.grad
            self.heatmaps.append(heatmap)
            self.bboxes.append(cls[...,:4].detach().cpu().numpy())
            
            