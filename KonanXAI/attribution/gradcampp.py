from KonanXAI.attribution import GradCAM
#from ..attribution import Attribution
#from ....utils import *
#from ....models import XAIModel
from KonanXAI.datasets import Datasets
#from ...core import darknet
import numpy as np
import cv2
import torch.nn.functional as F

class GradCAMpp(GradCAM):
    def __init__(self, model, dataset: Datasets, platform):
        super().__init__(model, dataset, platform)
    

    def calculate(self):
        self.get_feature_and_gradient()
        self.heatmaps = []
        #print('len(self.feature)', len(self.feature)) == 1
        for feature, gradient in zip(self.feature, self.gradient):
            b, ch, h, w = gradient.shape
            alpha = gradient.reshape(b, ch, -1).mean(2)
            weights = alpha.reshape(b, ch, 1, 1)
            heatmap = (weights * feature).sum(1)
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