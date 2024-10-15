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
            
        self.model.apply(backward_hook)

    def rescale_hook(self, module, grad_in, grad_out):
        threshold = 1e-7

        reference_x = module.baseline_in.pop()[0]
        x = module.input.pop()[0]
        delta_x = x - reference_x
        reference_y = module.baseline_out.pop()[0]
        y = module.output.pop()[0]
        delta_y = y - reference_y

        multiplier = (delta_y / (delta_x + threshold)).to(self.device)
        far_zero_x = (delta_x.abs() > threshold).float().to(self.device)
        multiplier_far_zero = far_zero_x * multiplier

        near_zero_x = (delta_x.abs() <= threshold).to(self.device)
        multiplier_near_zero = near_zero_x * multiplier

        multiplier = multiplier_near_zero + multiplier_far_zero
        multiplier = multiplier * delta_x

        
    
        return (multiplier,)
    
    def reveal_cancel_hook(self, module, grad_in, grad_out):
        reference_x = module.baseline_in.pop()[0]
        x = module.input.pop()[0]

        delta_x = x - reference_x
        delta_x_pos = ((delta_x >=0)).float().to(self.device) * delta_x
        delta_x_neg = ((delta_x < 0)).float().to(self.device) * delta_x



        #print(grad_in.shape)
        return grad_in
    
    def linear_hook(self, module, grad_in, grad_out):
        reference_x = module.baseline_in.pop()[0]
        x = module.input.pop()[0]
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

        reference_y = module.baseline_out.pop()[0]
        y = module.output[0]
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



        return (grad_in[0],) + (multiplier,) + grad_in[2:]
    

    
    # def linear_hook(self, module, grad_in, grad_out):
    #     print(module)
    #     reference_x = module.baseline_in.pop()[0].squeeze(0)
    #     x = module.input.pop()[0].squeeze(0)
    #     delta_x = x - reference_x
    #     delta_x_pos = (delta_x > 0).float().to(self.device) * delta_x
    #     delta_x_neg = (delta_x < 0).float().to(self.device) * delta_x
    #     delta_x_zero = (delta_x == 0).float().to(self.device)

    #     weight = module.weight.detach().clone().to(self.device)

    #     y = module.output.pop()[0].unsqueeze(0)
    #     reference_y = module.baseline_out.pop()[0].unsqueeze(0)
    #     delta_y = y - reference_y

    #     delta_y_pos = (delta_y >0).float().to(self.device)
    #     delta_y_neg = (delta_y <0).float().to(self.device)

    #     multiplier_pos_pos = F.linear(delta_y_pos * grad_out[0], module.weight.T)
    #     multiplier_neg_pos = F.linear(delta_y_pos * grad_out[0], module.weight.T)
    #     multiplier_pos_neg = F.linear(delta_y_neg * grad_out[0], module.weight.T)
    #     multiplier_neg_neg = F.linear(delta_y_neg * grad_out[0], module.weight.T)


    #     # 첫번째 방법
    #     # pos_mask = (torch.matmul(weight, delta_x)>0).float().to(self.device)
    #     # neg_mask = (torch.matmul(weight, delta_x)<0).float().to(self.device)

    #     # multiplier_pos_pos = pos_mask * torch.matmul(weight, delta_x_pos) 
    #     # multiplier_pos_pos = multiplier_pos_pos.unsqueeze(0).T
    #     # multiplier_pos_pos = torch.matmul(multiplier_pos_pos, (1/delta_x_pos).unsqueeze(0))
    #     # multiplier_neg_pos = pos_mask * torch.matmul(weight, delta_x_neg)
    #     # multiplier_neg_pos = multiplier_neg_pos.unsqueeze(0).T
    #     # multiplier_neg_pos = torch.matmul(multiplier_neg_pos, (1/delta_x_neg).unsqueeze(0))
    #     # multiplier_pos_neg = neg_mask * torch.matmul(weight, delta_x_pos)
    #     # multiplier_pos_neg = multiplier_pos_neg.unsqueeze(0).T
    #     # multiplier_pos_neg = torch.matmul(multiplier_pos_neg, (1/delta_x_pos).unsqueeze(0))
    #     # multiplier_neg_neg = neg_mask * torch.matmul(weight, delta_x_neg)
    #     # multiplier_neg_neg = multiplier_neg_neg.unsqueeze(0).T
    #     # multiplier_neg_neg = torch.matmul(multiplier_neg_neg, (1/delta_x_neg).unsqueeze(0))


    #     multiplier = multiplier_pos_pos + multiplier_neg_pos + multiplier_pos_neg + multiplier_neg_neg
    #     # multiplier = torch.matmul(multiplier.T, grad_out[0].T).squeeze(0).T
    

    #     return (grad_in[0],) + (multiplier,) + (grad_in[2],)

    def linear_conv_hook(self, module, grad_in, grad_out):
        reference_x = module.baseline_in.pop()[0]
        x = module.input.pop()[0]
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

        reference_y = module.baseline_out.pop()[0]
        y = module.output.pop()[0]
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

        if grad_in[0] == None:
            self.input.grad = multiplier
            print(self.input.grad)
        else:
            return (multiplier,) + grad_in[1:]

    
    # def linear_conv_hook(self, module, grad_in, grad_out):
    #     print(module)

    #     reference_x = module.baseline_in.pop()[0]
    #     x = module.input.pop()[0]
    #     delta_x = x - reference_x
    #     delta_x_pos = (delta_x > 0).float().to(self.device) * delta_x
    #     delta_x_neg = (delta_x < 0).float().to(self.device) * delta_x
    #     delta_x_zero = (delta_x == 0).float().to(self.device)

        

    #     pos_mask = (module(delta_x) > 0).float().to(self.device)
    #     neg_mask = (module(delta_x) < 0).float().to(self.device)

    #     # 두번째 방법
    #     y = module.output.pop()[0].unsqueeze(0)
    #     reference_y = module.baseline_out.pop()[0].unsqueeze(0)
    #     delta_y = y - reference_y

    #     delta_y_pos = (delta_y > 0).float().to(self.device) * delta_y
    #     delta_y_neg = (delta_y < 0).float().to(self.device) * delta_y

    #     _, _, H, W = grad_out[0].shape
    #     Hnew = (H-1) * module.stride[0] - 2*module.padding[0] +\
    #                 module.dilation[0]*(module.kernel_size[0]-1) +\
    #                 module.output_padding[0] +1
    #     Wnew = (W-1) * module.stride[1] - 2 * module.padding[1] +\
    #                 module.dilation[1] * (module.kernel_size[1]-1) +\
    #                 module.output_padding[1] + 1
    #     _, _, Hin, Win = x.shape
    #     multiplier_pos_pos = F.conv_transpose2d(delta_y_pos*grad_out[0], module.weight, bias = None, padding = module.padding,
    #                                        output_padding=(Hin-Hnew, Win - Wnew), stride = module.stride,
    #                                        dilation= module.dilation, groups = module.groups,).to(self.device)
    #     multiplier_neg_pos = F.conv_transpose2d(delta_y_pos*grad_out[0], module.weight, bias = None, padding = module.padding,
    #                                        output_padding=(Hin-Hnew, Win - Wnew), stride = module.stride,
    #                                        dilation= module.dilation, groups = module.groups,).to(self.device)
    #     multiplier_pos_neg = F.conv_transpose2d(delta_y_neg*grad_out[0], module.weight, bias = None, padding = module.padding,
    #                                        output_padding=(Hin-Hnew, Win - Wnew), stride = module.stride,
    #                                        dilation= module.dilation, groups = module.groups,).to(self.device)
    #     multiplier_neg_neg = F.conv_transpose2d(delta_y_neg*grad_out[0], module.weight, bias = None, padding = module.padding,
    #                                        output_padding=(Hin-Hnew, Win - Wnew), stride = module.stride,
    #                                        dilation= module.dilation, groups = module.groups,).to(self.device)
    

        

    #     # 첫번재 방법
    #     # multiplier_pos_pos = pos_mask * module(delta_x_pos)
    #     # multiplier_pos_pos = multiplier_pos_pos.transpose(0,1)
    #     # multiplier_pos_pos = torch.matmul(multiplier_pos_pos, 1/delta_x_pos)
    #     # multiplier_neg_pos = pos_mask * module(delta_x_neg)
    #     # multiplier_neg_pos = multiplier_neg_pos.transpose(0,1)
    #     # multiplier_neg_pos = torch.matmul(multiplier_neg_pos, 1/delta_x_neg)
    #     # multiplier_pos_neg = neg_mask * module(delta_x_pos)
    #     # multiplier_pos_neg = multiplier_pos_neg.transpose(0,1)
    #     # multiplier_pos_neg = torch.matmul(multiplier_pos_neg, 1/delta_x_pos)
    #     # multiplier_neg_neg = neg_mask * module(delta_x_neg)
    #     # multiplier_neg_neg = multiplier_neg_neg.transpose(0,1)
    #     # multiplier_neg_neg = torch.matmul(multiplier_neg_neg, 1/delta_x_neg)

    #     multiplier = multiplier_pos_pos + multiplier_neg_pos + multiplier_pos_neg + multiplier_neg_neg
    #     if grad_in[0] == None:
    #         self.input.grad = multiplier
    #         print(self.input.grad)
    #     else:
    #         return (multiplier_pos_pos,) + grad_in[1:]

    #     # new_multiplier = torch.zeros(grad_in[0].shape).to(self.device)
    #     # for i in range(multiplier.shape[2]):
    #     #     for j in range(multiplier.shape[3]):
    #     #         m = multiplier[:, :, i, j].transpose(0,1).squeeze(-1).squeeze(-1)
    #     #         g = grad_out[0][:,:,i,j].transpose(0,1).squeeze(-1).squeeze(-1)
    #     #         m = torch.matmul(m,g)
    #     #         new_multiplier[:, :, i, j] = m




    
    def calculate(self):
        self.set_baseline()
        self.set_baseline_forward_hook()
        
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
        
        contr_score = self.input.grad[0].unsqueeze(0)
        contr_score = torch.sum(contr_score, dim=1)




        return contr_score
    
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
            
        
 