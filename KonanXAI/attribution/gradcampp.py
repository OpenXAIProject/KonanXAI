from KonanXAI.attribution import GradCAM
#from ..attribution import Attribution
#from ....utils import *
#from ....models import XAIModel
from KonanXAI.datasets import Datasets
import darknet
import numpy as np
import cv2
import torch.nn.functional as F
import torch
class GradCAMpp(GradCAM):
    def calculate(self):
        self.get_feature_and_gradient()
        self.heatmaps = []
        for index, (feature, gradient) in enumerate(zip(self.feature, self.gradient)):
            b, ch, h, w = gradient.shape
            alpha_num = gradient.pow(2)
            alpha_denom = gradient.pow(2).mul(2) + \
                    feature.mul(gradient.pow(3)).view(b, ch, h*w).sum(-1, keepdim=True).view(b, ch, 1, 1)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alpha = alpha_num.div(alpha_denom+1e-7)
            if 'yolo' in self.model.model_name:
                self.positive_gradients = F.relu(self.logits[0][index][self.label_index[index]].exp()*gradient) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
            else:
                self.positive_gradients = F.relu(self.pred[0][self.label_index].exp()*gradient) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
            weights = (alpha*self.positive_gradients).view(b, ch, h*w).sum(-1).view(b, ch, 1, 1)
            heatmap = (weights * feature).sum(1, keepdim=True)
            heatmap = F.relu(heatmap)
            self.heatmaps.append(heatmap)
            
        if self.model_name[0:4] == 'yolo':
            return self.heatmaps, self.bboxes
        else:
            return self.heatmaps
            
    def _yolo_backward_darknet(self):
        net: darknet.Network = self.model.net
        self.gradcam = []
        # TODO - Target Layer 는 정해졌다고 가정
        # target_layer = [net.layers[138], net.layers[149], net.layers[160]]
        select_layer = set(list(self.bbox_layer.values()))
        target_layer = [net.layers[index-1] for index in select_layer]
        print(f"select_layer: {select_layer}")
        # target_layer = [net.layers[30], net.layers[37]]
        for box in self.bboxes:
            i = self.bbox_layer[box.entry]
            layer = net.layers[i]
            out = layer.get_output()
            net.zero_grad()
            # feature index
            stride = layer.out_w * layer.out_h
            idx = box.entry + (5 + box.class_idx) * stride
            # set delta
            layer.delta[idx] = out[idx]
            net.backward()
            # Get Features
            gradcam = []
            target = layer
            feature = np.array(target.get_output())
            gradient = np.array(target.get_delta())
            stride = target.out_w * target.out_h
                # Reshape
            feature = feature.reshape((-1, target.out_w, target.out_h))            
            gradient = gradient.reshape((-1, stride)).mean(1)
                #2계3계미분 근사화
            eq_numerator = np.power(gradient,2) #분자
            eq_denominator = np.multiply(np.power(gradient,2),2) + (np.multiply(feature.reshape(-1,stride).mean(1), np.power(gradient,3))+1e-7) #분모
            weight = eq_numerator / (np.where(eq_denominator != 0.0, eq_denominator,0))
            weight = weight.reshape(-1,1,1)
                # Append
            gradcam.append((feature, weight))
            self.gradcam.append(gradcam)