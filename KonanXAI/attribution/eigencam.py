from KonanXAI.attribution import GradCAM
#from ..attribution import Attribution
#from ....utils import *
#from ....models import XAIModel
from KonanXAI.datasets import Datasets
import darknet
import numpy as np
import cv2
import torch
class EigenCAM(GradCAM):
    def calculate(self):
        self.get_feature_and_gradient()
        self.heatmaps = []
        with torch.no_grad():
            for feature_map in self.feature:
                activation_batch = feature_map
                activation_batch[torch.isnan(activation_batch)] = 0
                for activations in activation_batch:
                    reshaped_activations = (activations).reshape(activations.shape[0], -1).T
                    if "yolo" in self.model_name.lower():
                        reshaped_activations_min,reshaped_activations_max = reshaped_activations.min(),reshaped_activations.max()
                        reshaped_activations = (reshaped_activations - reshaped_activations_min).div(reshaped_activations_max-reshaped_activations_min).data
                    else:
                        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
                    reshaped_activations = reshaped_activations.detach().cpu()
                    _, _, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
                    projection = reshaped_activations @ VT[0,:]
                    projection = projection.reshape(activations.shape[1:])
                self.heatmaps.append(projection.unsqueeze(0).unsqueeze(0))
        if self.model_name[0:4] == 'yolo':
            return self.heatmaps, self.bboxes
        else:
            return self.heatmaps
    
    def _yolo_backward_darknet(self):
        net: darknet.Network = self.model.net
        self.gradcam = []
        # target_layer = []
        # TODO - Target Layer 는 정해졌다고 가정
        select_layer = set(list(self.bbox_layer.values()))
        target_layer = [net.layers[index-1] for index in select_layer]
        print(f"select_layer: {select_layer}")
        gradcam = []
        for target in target_layer:
            feature = np.array(target.get_output())
            # Reshape
            feature = feature.reshape((-1, target.out_w, target.out_h))            
            activations = feature
            activations[np.isnan(activations)] = 0
            reshaped_activations = (activations).reshape(activations.shape[0], -1).T
            reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
            _, _, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
            projection = reshaped_activations @ VT[0,:]
            projection = projection.reshape(activations.shape[1:])
            # Append
            gradcam.append((projection))
        self.gradcam.append(gradcam)