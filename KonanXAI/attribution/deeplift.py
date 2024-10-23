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
        self.rule = config['rule']
        self.forward_baseline_hooks = []
        self.forward_hooks = []
        self.backward_hooks = []
        self.remove_attr_hooks = []
                        
    def set_baseline(self):
        if self.baseline == 'zero':
            self.baseline = torch.zeros(self.input.shape).to(self.device)
        
    def set_baseline_forward_hook(self):
        def forward_hook(layer):
            self.forward_baseline_hooks.append(layer.register_forward_hook(self.baseline_forward_hook))
        self.model.apply(forward_hook)


    def baseline_forward_hook(self, module, input, output):
        if hasattr(module, 'index_list') == False:
            module.index_list = []
        if hasattr(module, 'index') == False:
            module.index = 0
            module.index_list.append(module.index)
        else:
            module.index = module.index + 1
            module.index_list.append(module.index)

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
                if self.rule == 'rescale':
                    self.backward_hooks.append(layer.register_backward_hook(self.rescale_hook))
                elif self.rule == 'reveal-cancel':
                    self.backward_hooks.append(layer.register_backward_hook(self.reveal_cancel_hook))   
            elif isinstance(layer, torch.nn.Linear):
                self.backward_hooks.append(layer.register_backward_hook(self.linear_hook))
            elif isinstance(layer, torch.nn.Conv2d):
                self.backward_hooks.append(layer.register_backward_hook(self.linear_conv_hook))
            
        self.model.apply(backward_hook)


   


    def rescale_hook(self, module, grad_in, grad_out):
        threshold = 1e-7

        reference_x = module.baseline_in[module.index][0]
        x = module.input[module.index][0]
        delta_x = x - reference_x
        reference_y = module.baseline_out[module.index][0]
        y = module.output[module.index][0]
        delta_y = y - reference_y

        multiplier = (delta_y / (delta_x + threshold)).to(self.device)
        far_zero_x = (delta_x.abs() > threshold).float().to(self.device)
        multiplier_far_zero = far_zero_x * multiplier

        near_zero_x = (delta_x.abs() <= threshold).to(self.device)
        multiplier_near_zero = near_zero_x * multiplier

        multiplier = multiplier_near_zero + multiplier_far_zero
        multiplier = multiplier * delta_x
        module.index = module.index -1
        if module.index not in module.index_list:
            module.index = module.index_list[-1]
    
        return (multiplier,)
    
    def reveal_cancel_hook(self, module, grad_in, grad_out):
        reference_x = module.baseline_in[module.index][0]
        x = module.input[module.index ][0]

        delta_x = x - reference_x
        delta_x_pos = ((delta_x >=0)).float().to(self.device) * delta_x
        delta_x_neg = ((delta_x < 0)).float().to(self.device) * delta_x

        delta_y_pos = 0.5 * (F.relu(reference_x + delta_x_pos) - F.relu(reference_x)) +\
                        0.5 * (F.relu(reference_x + delta_x_pos + delta_x_neg) - F.relu(reference_x + delta_x_neg))
        delta_y_neg = 0.5 * (F.relu(reference_x + delta_x_neg) - F.relu(reference_x)) +\
                        0.5 * (F.relu(reference_x + delta_x_pos + delta_x_neg) - F.relu(reference_x + delta_x_pos))

        multiplier_pos = delta_y_pos / (delta_x_pos + 1e-7)
        multiplier_neg = delta_y_neg / (delta_x_neg + 1e-7)

        grad_out_pos = ((grad_out[0] >= 0).float().to(self.device)) * grad_out[0]
        grad_out_neg = ((grad_out[0] < 0).float().to(self.device)) * grad_out[0]

        multiplier = grad_out_pos * multiplier_pos + grad_out_neg * multiplier_neg
        module.index = module.index -1
        if module.index not in module.index_list:
            module.index = module.index_list[-1]
        return (multiplier,)
    
    def linear_hook(self, module, grad_in, grad_out):
        reference_x = module.baseline_in[module.index][0]
        x = module.input[module.index][0]
        delta_x = x - reference_x
        delta_x_pos = ((delta_x>0).float().to(self.device))
        delta_x_neg = ((delta_x<0).float().to(self.device))
        delta_x_zero = ((delta_x == 0).float().to(self.device))

        transposed_weight = module.weight.detach().clone().T.to(self.device)
        size = transposed_weight.shape
        transpose_pos = nn.Linear(size[1], size[0]).to(self.device)
        transpose_pos.weight = nn.Parameter(((transposed_weight >0).float().to(self.device))*transposed_weight)
        transpose_neg = nn.Linear(size[1],size[0]).to(self.device)
        transpose_neg.weight = nn.Parameter(((transposed_weight <0).float().to(self.device))*transposed_weight)

        transpose_full = nn.Linear(size[1], size[0]).to(self.device)
        transpose_full.weight = nn.Parameter(transposed_weight)

        reference_y = module.baseline_out[module.index][0]
        y = module.output[module.index][0]
        delta_y = y - reference_y
        delta_y_pos = ((delta_y >0).float().to(self.device))*delta_y
        delta_y_neg = ((delta_y <0).float().to(self.device))*delta_y

        pos_grad_out = delta_y_pos * grad_out[0]
        neg_grad_out = delta_y_neg * grad_out[0]

        pos_pos_result = transpose_pos.forward(pos_grad_out) *delta_x_pos
        pos_neg_result = transpose_pos.forward(neg_grad_out) * delta_x_pos
        neg_pos_result = transpose_neg.forward(neg_grad_out) *delta_x_neg
        neg_neg_result = transpose_neg.forward(neg_grad_out) * delta_x_neg
        null_result = transpose_full.forward(grad_out[0]) * delta_x_zero

        multiplier = pos_pos_result + pos_neg_result + neg_pos_result + neg_neg_result + null_result
        module.index = module.index -1
        if module.index not in module.index_list:
            module.index = module.index_list[-1]


        return (grad_in[0],) + (multiplier,) + grad_in[2:]
    

    def linear_conv_hook(self, module, grad_in, grad_out):
        reference_x = module.baseline_in[module.index][0]
        x = module.input[module.index][0]
        delta_x = x - reference_x
        delta_x_pos = ((delta_x>0).float().to(self.device))
        delta_x_neg = ((delta_x<0).float().to(self.device))
        delta_x_zero = ((delta_x == 0).float().to(self.device))

        transpose_pos = nn.ConvTranspose2d(module.out_channels, module.in_channels, module.kernel_size,
                                           module.stride, module.padding).to(self.device)
        transpose_pos.weight = nn.Parameter(((module.weight>0).float().to(self.device)) * module.weight.detach().clone().to(self.device)).to(self.device)
        transpose_neg = nn.ConvTranspose2d(module.out_channels, module.in_channels, module.kernel_size,
                                           module.stride, module.padding).to(self.device)
        transpose_neg.weight = nn.Parameter(((module.weight<0).float().to(self.device)) * module.weight.detach().clone().to(self.device)).to(self.device)
        transpose_full = nn.ConvTranspose2d(module.out_channels, module.in_channels, module.kernel_size,
                                           module.stride, module.padding).to(self.device)
        transpose_full.weight = nn.Parameter((module.weight.detach().clone().to(self.device))).to(self.device)

        reference_y = module.baseline_out[module.index][0]
        y = module.output[module.index][0]
        delta_y = (y-reference_y).to(self.device)
        delta_y_pos = ((delta_y >0).float().to(self.device)) * delta_y
        delta_y_neg = ((delta_y <0)).float().to(self.device) * delta_y

        pos_grad_out = delta_y_pos * grad_out[0]
        neg_grad_out = delta_y_neg * grad_out[0]

        dim_check = transpose_pos.forward(pos_grad_out)

        if dim_check.shape != delta_x.shape:
            if dim_check.shape[3] > delta_x.shape[3]:
                dim_diff = dim_check.shape[3] - delta_x.shape[3]
                delta_x = torch.cat((delta_x, torch.ones(delta_x.shape[0], delta_x.shape[1], dim_diff, delta_x.shape[3]).to(self.device)), 2)
                delta_x = torch.cat((delta_x, torch.ones(delta_x.shape[0], delta_x.shape[1], delta_x.shape[2], dim_diff).to(self.device)),3)
            else:
                new_shape = dim_check.shape
                delta_x = delta_x[0:new_shape[0], 0:new_shape[1], 0:new_shape[2], 0:new_shape[3]]

            delta_x_pos = ((delta_x>0).float().to(self.device))
            delta_x_neg = ((delta_x <0).float().to(self.device))
            delta_x_zero = ((delta_x == 0).float().to(self.device))

        pos_pos_result = transpose_pos.forward(pos_grad_out) * delta_x_pos
        pos_neg_result = transpose_pos.forward(neg_grad_out) * delta_x_pos
        neg_pos_result = transpose_neg.forward(neg_grad_out) * delta_x_neg
        neg_neg_result = transpose_neg.forward(pos_grad_out) * delta_x_neg
        null_result = transpose_full.forward(grad_out[0]) * delta_x_zero

        multiplier = pos_pos_result + pos_neg_result + neg_pos_result + neg_neg_result + null_result
        
        if x.shape != multiplier.shape:
            if x.shape[3] > multiplier.shape[3]:
                dim_diff = x.shape[3] - multiplier.shape[3]
                multiplier = torch.cat((multiplier, torch.ones(multiplier.shape[0], multiplier.shape[1], dim_diff, multiplier.shape[3]).to(self.device)),2)
                multiplier = torch.cat((multiplier, torch.ones(multiplier.shape[0], multiplier.shape[1], multiplier.shape[2], dim_diff).to(self.device)), 3)
            else:
                new_shape = x.shape
                multiplier = delta_x[0:new_shape[0], 0:new_shape[1], 0:new_shape[2], 0:new_shape[3]]
        module.index = module.index -1
        if module.index not in module.index_list:
            module.index = module.index_list[-1]
        if grad_in[0] == None:
            self.input.grad = multiplier
    
        else:
            return (multiplier,) + grad_in[1:]

  

    
    def calculate(self):
        if self.framework == 'torch':
            

            if self.model_name in ('yolov4', 'yolov4-tiny', 'yolov5s'):
                self.set_baseline()
                self._yolo_get_bbox_pytorch()
                self._yolo_backward_pytorch()
                
                
                for handle in self.forward_hooks:
                    handle.remove()

                for handle in self.backward_hooks:
                    handle.remove()

                # for module in self.model.modules():
                #     print(module)
                #     del module.input
                #     del module.output
                #     del module.baseline_in
                #     del module.baseline_out
                #     del module.index
                #     del module.index_list
                

                return self.contr_scores, self.bboxes

            
            else:      
                self.set_baseline()
                self.set_baseline_forward_hook()
                if self.input.grad != None:
                    self.input.grad.zero_()
                delta_x = self.input - self.baseline
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

                for handle in self.backward_hooks:
                    handle.remove()
                
                contr_score = self.input.grad[0].unsqueeze(0).clone().detach()
                contr_score = torch.sum(contr_score, dim=1).unsqueeze(0)
                
                for module in self.model.modules():
                    print(module)
                    del module.input
                    del module.output
                    del module.baseline_in
                    del module.baseline_out
                    del module.index
                    del module.index_list

                return contr_score
        
        elif self.framework == 'darknet':
            pass
    
        
    def _yolo_get_bbox_pytorch(self):
        self.set_baseline_forward_hook()
        self.input.requires_grad=True
        self.baseline_preds, baseline_raw_logit = self.model(self.input)
        for handle in self.forward_baseline_hooks:
            handle.remove()
        self.set_forward_hook()
        self.set_backward_hook()
        for param in self.model.parameters():
            param.requires_grad = True
        self.preds_origin, raw_logit = self.model(self.input)
        

        self.logits_origin = torch.concat([data.view(-1,self.preds_origin.shape[-1])[...,5:] for data in raw_logit],dim=0)
        self.baseline_logits_origin = torch.concat([data.view(-1,self.baseline_preds.shape[-1])[...,5:] for data in baseline_raw_logit],dim=0)

        with torch.no_grad():
            self.preds, logits, self.select_layers = non_max_suppression(self.preds_origin, self.logits_origin.unsqueeze(0), conf_thres=0.25, model_name = self.model_name)
        self.index_tmep = yolo_choice_layer(raw_logit, self.select_layers)
        
    def _yolo_backward_pytorch(self):
        self.bboxes = []
        self.contr_scores = []
        for cls, sel_layer, sel_layer_index in zip(self.preds[0], self.select_layers, self.index_tmep):
            self.model.zero_grad()
            delta = self.logits_origin - self.baseline_logits_origin
            gradient = torch.zeros(self.logits_origin.shape).to(self.device)
            gradient[sel_layer][int(cls[5].item())] = delta[sel_layer][int(cls[5].item())]
            if self.input.grad != None:
                self.input.grad.zero_()
            self.logits_origin.backward(gradient, retain_graph=True)
            #self.logits_origin[sel_layer][int(cls[5].item())].backward(gradient)
            
            contr_score = self.input.grad[0].unsqueeze(0).clone().detach()
            contr_score = torch.sum(contr_score, dim=1).unsqueeze(0)
            self.contr_scores.append(contr_score)
            self.bboxes.append(cls[...,:4].detach().cpu().numpy())