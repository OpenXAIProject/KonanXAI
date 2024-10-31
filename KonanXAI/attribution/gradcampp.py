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
__all__ = ["GradCAMpp"]
class GradCAMpp(GradCAM):
    def calculate(self, inputs = None, targets = None):
        self.get_feature_and_gradient(inputs, targets)
        self.heatmaps = []
        for index, (feature, gradient) in enumerate(zip(self.feature, self.gradient)):
            b, ch, h, w = gradient.shape
            alpha_num = gradient.pow(2)
            alpha_denom = gradient.pow(2).mul(2) + \
                    feature.mul(gradient.pow(3)).view(b, ch, h*w).sum(-1, keepdim=True).view(b, ch, 1, 1)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alpha = alpha_num.div(alpha_denom+1e-7)
            if 'yolo' in self.model.model_name and self.framework =="torch":
                self.positive_gradients = F.relu(self.logits[0][index][self.label_index[index]].exp()*gradient) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
            elif 'yolo' in self.model.model_name and self.framework == "darknet":
                self.positive_gradients = F.relu(self.logits.exp()*gradient)
            else:
                if self.framework == 'dtrain':
                    self.positive_gradients = F.relu(np.exp(self.preds)*gradient)
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
            