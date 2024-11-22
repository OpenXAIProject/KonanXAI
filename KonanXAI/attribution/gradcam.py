from KonanXAI._core.pytorch.yolov5s.utils import non_max_suppression, yolo_choice_layer
#from ..attribution import Attribution
from KonanXAI._core.pytorch.anchorFreeYolo.utils import anchor_free_non_max_suppression, anchor_free_yolo_choice_layer
from KonanXAI.attribution.gradient import Gradient
from KonanXAI.utils import *
#from ....models import XAIModel
from KonanXAI.datasets import Datasets
#import darknet 
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from KonanXAI._core.dtrain.utils import convert_tensor_to_numpy

__all__ = ["GradCAM"]
class GradCAM(Gradient):        
    def calc_dtrain(self):
        
        forward_args = {
            "train_mode": True
        }
        backward_args = {
            "hook": [],
        }
        for key in self.target_layer:
            hook = {"layer": key, "roles": ["x", "y", "xgrad", "ygrad"]}
            backward_args["hook"].append(hook)
        # Convert DtrainTensor
        if isinstance(self.input, torch.Tensor):
            ndarray = self.input.cpu().detach().numpy()
            self.input = self.model.session.createTensorFromArray(ndarray)
        elif isinstance(self.input, np.ndarray):
            self.input = self.model.session.createTensorFromArray(self.input)
        tensor_dict = {}
        x = {"#": self.input}
        self.model.execute_forward(x, forward_args, tensor_dict)
        self.pred = tensor_dict['pred']
        self.label_index = tensor_dict['pred'].argmax(1).value()
        self.preds = convert_tensor_to_numpy(self.pred)[0][self.label_index]
        # probs = tensor_dict['pred'].softmax(1)
        print("Pred :", self.label_index)
        y = self.model.session.createTensorZeros(self.pred.shape)
        y[self.label_index] = 1.0
        y = {"#": y}
        tensor_dict = {}
        self.model.execute_backward(y, backward_args, tensor_dict)
        feature_map = tensor_dict[self.target_layer[-1] + ".y"]
        class_grad = tensor_dict[self.target_layer[-1] + ".ygrad"]
        feature_map = convert_tensor_to_numpy(feature_map)
        class_grad = convert_tensor_to_numpy(class_grad)

        feature_map = torch.tensor(feature_map).unsqueeze(0)
        class_grad = torch.tensor(class_grad).unsqueeze(0)
        return feature_map, class_grad
        
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
                    if "act" in layer:
                        base_layer.act = torch.nn.SiLU(inplace=False)
                    base_layer = base_layer._modules[layer]
                self.layer.append(base_layer)

    def set_model_hook(self):    
        if 'yolo' in self.model_name:
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
    
    def get_feature_and_gradient(self, inputs, targets):
        if inputs != None:
            self.input = inputs
        if targets != None:
            self.label_index = targets
        self.feature = []
        self.gradient = []
        if self.framework == 'torch':
            self.input.requires_grad = True
            self._get_target_layer()
            fwd_handle, bwd_handle = self.set_model_hook()
            
            if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                self.model.eval()
                self._yolo_get_bbox_pytorch()
                self._yolo_backward_pytorch()
                
            elif "af_yolo" in self.model_name:
                self._anchor_free_yolo_get_bbox_pytorch()
                self._yolo_backward_pytorch()

            else:
                self.model.eval()
                if self.model.model_algorithm == 'abn':
                    self.att, self.pred, _ = self.model(self.input)
                else:
                    self.pred = self.model(self.input)
                if self.label_index == None:
                    self.label_index = torch.argmax(self.pred).item()
                self.pred[0][self.label_index].backward()
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

        elif self.framework == 'dtrain':
            self.feature, self.gradient = self.calc_dtrain()
            return self.feature, self.gradient
            
    def calculate(self, inputs=None, targets = None):
        self.get_feature_and_gradient(inputs, targets)
        self.heatmaps = [] 
        for feature, gradient in zip(self.feature, self.gradient):
            b, ch, h, w = gradient.shape
            alpha = gradient.reshape(b, ch, -1).mean(2)
            weights = alpha.reshape(b, ch, 1, 1)
            heatmap = (weights * feature).sum(1, keepdim=True)
            heatmap = F.relu(heatmap)
            self.heatmaps.append(heatmap)
            
        if 'yolo' in self.model_name:
            return self.heatmaps, self.bboxes
        else:
            return self.heatmaps
            
    def _yolo_get_bbox_pytorch(self):
        self.pred_origin, raw_logit = self.model(self.input)
        self.logits_origin = torch.concat([data.view(-1,self.pred_origin.shape[-1])[...,5:] for data in raw_logit],dim=0)
        with torch.no_grad():
            self.pred, self.logits, self.select_layers = non_max_suppression(self.pred_origin, self.logits_origin.unsqueeze(0), conf_thres=0.25, model_name = self.model_name)
        self.index_tmep = yolo_choice_layer(raw_logit, self.select_layers)
        
    def _anchor_free_yolo_get_bbox_pytorch(self):
        self.pred_origin, raw_logit = self.model(self.input)
        shape = raw_logit[0].shape
        reg_max = self.model.model[-1].reg_max
        x_cat = torch.cat([xi.view(shape[0], shape[1], -1) for xi in raw_logit], 2)
        grid_cell, self.class_prob = x_cat.split((reg_max*4, self.model.nc), 1)
        self.logits_origin = self.class_prob.squeeze(0).transpose(-1,-2)
        with torch.no_grad():
            self.pred, self.logits , self.select_layers = anchor_free_non_max_suppression(self.pred_origin, self.logits_origin.unsqueeze(0).transpose(-1,-2), conf_thres=0.25)
        self.index_tmep = anchor_free_yolo_choice_layer(raw_logit, self.select_layers)
        
    def _yolo_backward_pytorch(self):
        self.bboxes = []
        self.label_index = []
        for cls, sel_layer, sel_layer_index in zip(self.pred[0], self.select_layers, self.index_tmep):
            self.model.zero_grad()
            self.logits_origin[sel_layer][int(cls[5].item())].backward(retain_graph=True)
            layer = self.layer[sel_layer_index]
            feature = layer.fwd_out[-1].unsqueeze(0)
            gradient = layer.bwd_out[0]
            self.feature.append(feature)
            self.gradient.append(gradient)
            self.label_index.append(int(cls[5].item()))
            self.bboxes.append(cls[...,:4].detach().cpu().numpy())
    
    # def _yolo8_backward_pytorch(self):
    #     self.bboxes = []
    #     self.label_index = []
    #     for cls, sel_layer, sel_layer_index in zip(self.pred[0], self.select_layers, self.index_tmep):
    #         self.model.zero_grad()
    #         score = self.clas_prob[sel_layer][int(cls[5].item())]
    #         score.backward(retain_graph=True)
    #         if len(self.layer)>3:
    #             layer_cv2 = self.layer[sel_layer_index]
    #             layer_cv3 = self.layer[sel_layer_index+3]
    #             feature = torch.cat([layer_cv2.fwd_out[-1].unsqueeze(0),layer_cv3.fwd_out[-1].unsqueeze(0)],dim=1)
    #             gradient = torch.cat([layer_cv2.bwd_out[0],layer_cv3.bwd_out[0]],dim=1)
    #         else:
    #             layer = self.layer[sel_layer_index]
    #             feature = layer.fwd_out[-1].unsqueeze(0)
    #             gradient = layer.bwd_out[0]
    #         self.feature.append(feature)
    #         self.gradient.append(gradient)
    #         self.label_index.append(int(cls[5].item()))
    #         self.bboxes.append(cls[...,:4].detach().cpu().numpy())