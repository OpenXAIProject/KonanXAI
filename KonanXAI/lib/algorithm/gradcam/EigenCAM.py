from ..gradcam import GradCAM
from ..algorithm import Algorithm
from ....utils import *
from ....models import XAIModel
from ....datasets import Datasets
from ...core import darknet
import numpy as np
import cv2

class EigenCAM(GradCAM):
    def __init__(self, model: XAIModel, dataset: Datasets, platform: PlatformType):
        super().__init__(model, dataset, platform)
    
    def _norm_heatmap(self, heatmap):
        smin, smax = heatmap.min(), heatmap.max()
        if (smax - smin != 0):
            saliency = (heatmap - smin) / (smax - smin)
        else:
            saliency = heatmap
        return saliency
    
    def scale_image(self,cam, target_size=None):
            img = cam - np.min(cam)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result = np.float32(img)
            return result
    
    def _get_heatmap(self,feature,size=(608,608)):
        cam_per_target_layer = []
        cam = np.maximum(feature, 0)
        scaled = self.scale_image(cam, target_size=size)
        # cam_per_target_layer.append(scaled[None,:])
        # cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        # cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        # result = np.expand_dims(np.mean(cam_per_target_layer, axis=0),axis=0)
        # saliency = scale_image(result)
        # saliency = saliency[0,:,:]
        # saliency = self._norm_heatmap(saliency)
        return scaled#saliency

    def _gradcam(self):
        heatmaps = []
        saliency = None
        for gradcam in self.gradcam:
            maps = []
            if isinstance(gradcam, list):
                # Multi Layer GradCAM
                for index, feature in enumerate(gradcam):
                    heatmap = self._get_heatmap(feature)
                    maps.append(heatmap[None,:])
              
                cam_per_target_layer = np.concatenate(maps, axis=0)
                # cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
                result = np.sum(cam_per_target_layer, axis=0)
                saliency = self.scale_image(result,target_size=(608,608))
                saliency = saliency#[0,:,:]
                saliency = self._norm_heatmap(saliency)
                    # if index == 0:
                    #     saliency = heatmap
                    # else:
                    #     saliency = np.where(saliency < heatmap, heatmap, saliency)
                # Heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
            else:
                feature, weight = gradcam
                heatmap = self._get_heatmap(feature, weight)
            heatmaps.append(heatmap)
        return heatmaps
    
    def _yolo_backward_darknet(self):
        net: darknet.Network = self.model.net
        self.gradcam = []
        # target_layer = []
        # TODO - Target Layer 는 정해졌다고 가정
        select_layer = set(list(self.bbox_layer.values()))
        target_layer = [net.layers[index-1] for index in select_layer]
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