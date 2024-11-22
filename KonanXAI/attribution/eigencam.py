from KonanXAI.attribution import GradCAM
#from ..attribution import Attribution
#from ....utils import *
#from ....models import XAIModel
from KonanXAI.datasets import Datasets
#import darknet
import numpy as np
import cv2
import torch
__all__ = ["EigenCAM"]
class EigenCAM(GradCAM):
    def calculate(self,inputs=None,targets=None):
        self.get_feature_and_gradient(inputs,targets)
        self.heatmaps = []
        with torch.no_grad():
            for feature_map in self.feature:
                activation_batch = feature_map
                activation_batch[torch.isnan(activation_batch)] = 0
                for activations in activation_batch:
                    reshaped_activations = (activations).reshape(activations.shape[0], -1).T
                    if "yolo" in self.model_name:
                        reshaped_activations_min,reshaped_activations_max = reshaped_activations.min(),reshaped_activations.max()
                        reshaped_activations = (reshaped_activations - reshaped_activations_min).div(reshaped_activations_max-reshaped_activations_min).data
                    else:
                        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
                    reshaped_activations = reshaped_activations.detach().cpu()
                    _, _, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
                    projection = reshaped_activations @ VT[0,:]
                    projection = projection.reshape(activations.shape[1:])
                self.heatmaps.append(projection.unsqueeze(0).unsqueeze(0))
        if 'yolo' in self.model_name:
            return self.heatmaps, self.bboxes
        else:
            return self.heatmaps
