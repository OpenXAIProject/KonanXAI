from KonanXAI._core.pytorch.yolov5s.utils import non_max_suppression, yolo_choice_layer
#from ..attribution import Attribution
from KonanXAI.attribution.layer_wise_propagation.lrp_tracer import Graph
from KonanXAI.utils import *
#from ....models import XAIModel
from KonanXAI.datasets import Datasets
import darknet 
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import copy
import torch.nn as nn

# Attribution 상속 지음
# yolo target_layer = [model, '23', 'cv1','conv']
class DeepLIFT:
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
        self.yaml_path = None
        if framework.lower() == "darknet":
            self.input = input
            self.input_size = self.input.shape
        else:
            self.input = input[0].to(self.device)
            self.input_size = self.input.shape[2:4]
        self.baseline = config['baseline']
        self.target_class = config['target_class']
        self.forward_baseline_hooks = []
        self.forward_hooks = []
        self.backward_hooks = []
                        
    def set_baseline(self):
        if self.baseline == 'zero':
            self.baseline = torch.zeros(self.input.shape).to(self.device)
        
    def set_baseline_forward_hook(self):
        def forward_hook(layer):
            self.forward_baseline_hooks.append(layer.register_forward_hook(self.baseline_forward_hook))
        self.model.apply(forward_hook)


    def baseline_forward_hook(self, module, input, output):
        if 'baseline_in' not in dir(module):
            module.baseline_in = [input]
            module.baseline_out = [output]
        else:
            module.baseline_in.append(input)
            module.baseline_out.append(output)
 
    def set_forward_hook(self):
        def forward_hook(layer):
            self.forward_hooks.append(layer.register_forward_hook(self.forward_hook))
        self.model.apply(forward_hook)

    def forward_hook(self, module, input, output):
        if 'input' not in dir(module):
            module.input = [input]
            module.output = [output]
        else:
            module.input.append(input)
            module.output.append(output)
        
    
    def set_backward_hook(self):
        def backward_hook(layer):
            if isinstance(layer, torch.nn.ReLU):
                self.backward_hooks.append(layer.register_backward_hook(self.rescale_hook))
            elif isinstance(layer, torch.nn.Linear):
                self.backward_hooks.append(layer.register_backward_hook(self.linear_hook))
            elif isinstance(layer, torch.nn.Conv2d):
                self.backward_hooks.append(layer.register_backward_hook(self.linear_conv_hook))
            else:
                self.backward_hooks.append(layer.register_backward_hook(self.identity_hook))
        self.model.apply(backward_hook)

    def rescale_hook(self, module, grad_in, grad_out):
        threshold = 1e-7
        reference_x = module.baseline_in.pop()[0]
        x = module.input.pop()[0]
        delta_x = x - reference_x

        delta_y = grad_out[0]

        multiplier = (delta_y / (delta_x + threshold)).to(self.device)
        far_zero_x = (delta_x.abs() > threshold).float().to(self.device)
        conbribution_score_far_zero = far_zero_x * multiplier

        near_zero_x = (delta_x.abs() <= threshold).to(self.device)
        contribution_score_near_zero = near_zero_x * multiplier

        contribution_score = contribution_score_near_zero + conbribution_score_far_zero
        print(contribution_score.shape)
        print(grad_in[0].shape)
    
        return grad_in
    
    def reveal_calcel_hook(self, module, grad_in, grad_out):
        print(module)
        return grad_in
    
    def linear_hook(self, module, grad_in, grad_out):
        print(module)
        return grad_in
    
    def linear_conv_hook(self, module, grad_in, grad_out):
        print(module)
        return grad_in
    
    def identity_hook(self, module, grad_in, grad_out):
        print(module)
        return grad_in
    
    def calculate(self):
        self.set_baseline()
        self.set_baseline_forward_hook()
        
        
        
        self.model.eval()
        self.baseline = self.model(self.baseline)

        for handle in self.forward_baseline_hooks:
            handle.remove()
        

        self.set_forward_hook()
        self.set_backward_hook()
        if self.model.model_algorithm == 'abn':
            attr, pred, _ = self.model(self.input)
        else:
            pred = self.model(self.input)
        
        for handle in self.forward_hooks:
            handle.remove()

        
        

        self.model.zero_grad()
        delta = pred - self.baseline
        index = pred.argmax().item()
        gradient = torch.zeros(pred.shape).to(self.device)
        gradient[0][index] = delta[0][index]
        pred.backward(gradient)
        




        return self.input.grad
    
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
            
        
 