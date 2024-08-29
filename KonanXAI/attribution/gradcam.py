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
class GradCAM:
    def __init__(
            self, 
            framework, 
            model, 
            input, 
            config):
        '''
        input: [batch, channel, height, width] torch.Tensor 
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.framework = framework
        self.model = model
        self.model_name = self.model.model_name
        if framework.lower() == "darknet":
            self.input = input
            self.input_size = self.input.shape
        else:
            self.input = input[0].to(device)
            self.input_size = self.input.shape[2:4]
                        
        self.target_layer = config['target_layer']

    def _get_target_layer(self):
        if isinstance(self.target_layer, list):
            self.layer = self.model._modules[self.target_layer[0]]
            for layer in self.target_layer[1:]:
                self.layer = self.layer._modules[layer]

        elif isinstance(self.target_layer, dict):
            self.layer = []
            for key, layers in self.target_layer.items():
                base_layer = self.model._modules[layers[0]]
                for layer in layers[1:]:
                    base_layer = base_layer._modules[layer]
                self.layer.append(base_layer)

    
    def set_model_hook(self):    
        if 'yolo' in self.model_name[0:4]:
            fwd_handle, bwd_handle = [],[]
            for layer in self.layer:
                layer.fwd_in = []
                layer.fwd_out = []
                layer.bwd_in = []
                layer.bwd_out = []
                fwd_handle.append(layer.register_forward_hook(self._fwd_hook))
                bwd_handle.append(layer.register_full_backward_hook(self._bwd_hook))
        else: 
            self.layer.fwd_in = []
            self.layer.fwd_out = []
            self.layer.bwd_in = []
            self.layer.bwd_out = []
            fwd_handle = self.layer.register_forward_hook(self._fwd_hook)
            bwd_handle = self.layer.register_full_backward_hook(self._bwd_hook)
        return fwd_handle, bwd_handle
        
    
    def _fwd_hook(self, l, fx, fy):
        l.fwd_in.append(fx[0])
        l.fwd_out.append(fy[0])
        
    def _bwd_hook(self, l, bx, by):
        l.bwd_in.insert(0, bx[0])
        l.bwd_out.insert(0, by[0])
    
    def get_feature_and_gradient(self):
        self.feature = []
        self.gradient = []
        if self.framework == 'torch':
            self.model.eval()
            self.input.requires_grad = True
            self._get_target_layer()
            fwd_handle, bwd_handle = self.set_model_hook()
            
            if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                self._yolo_get_bbox_pytorch()
                self._yolo_backward_pytorch()

            else:
                if self.model.model_algorithm == 'abn':
                    self.att, self.pred, _ = self.model(self.input)
                else:
                    self.pred = self.model(self.input)
                label_index = torch.argmax(self.pred).item()
                self.pred[0][label_index].backward()
                feature = self.layer.fwd_in[-1]
                gradient = self.layer.bwd_in[-1]
                self.feature.append(feature)
                self.gradient.append(gradient)
                fwd_handle.remove()
                bwd_handle.remove()
                return self.feature, self.gradient

        elif self.framework == 'darknet':
            self.model.forward_image(self.input)
            self._yolo_get_bbox_darknet()
            self._yolo_backward_darknet()
            return self.feature, self.gradient

    
    def calculate(self):
        self.get_feature_and_gradient()
        self.heatmaps = []
        #print('len(self.feature)', len(self.feature)) == 1
        for feature, gradient in zip(self.feature, self.gradient):
            b, ch, h, w = gradient.shape
            alpha = gradient.reshape(b, ch, -1).mean(2)
            weights = alpha.reshape(b, ch, 1, 1)
            heatmap = (weights * feature).sum(1, keepdim=True)
            heatmap = F.relu(heatmap)
            self.heatmaps.append(heatmap)
            
        if self.model_name[0:4] == 'yolo':
            return self.heatmaps, self.bboxes
        else:
            return self.heatmaps
    
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
        self.preds_origin, raw_logit = self.model(self.input)
        self.logits_origin = torch.concat([data.view(-1,self.preds_origin.shape[-1])[...,5:] for data in raw_logit],dim=0)
        with torch.no_grad():
            self.preds, logits, self.select_layers = non_max_suppression(self.preds_origin, self.logits_origin.unsqueeze(0), conf_thres=0.25, model_name = self.model_name)
        self.index_tmep = yolo_choice_layer(raw_logit, self.select_layers)
        
    def _yolo_backward_pytorch(self):
        self.bboxes = []
        for cls, sel_layer, sel_layer_index in zip(self.preds[0], self.select_layers, self.index_tmep):
            self.model.zero_grad()
            self.logits_origin[sel_layer][int(cls[5].item())].backward(retain_graph=True)
            layer = self.layer[sel_layer_index]
            
            # 여기서는 fwd_out, bwd_out을 썼네?
            # feature, gradient, cls[...:4] 에서 .detach().cpu().numpy()를 써야할 이유가 있나?
            feature = layer.fwd_out[-1].unsqueeze(0)
            gradient = layer.bwd_out[0]
            self.feature.append(feature)
            self.gradient.append(gradient)
            self.bboxes.append(cls[...,:4].detach().cpu().numpy())
        
 