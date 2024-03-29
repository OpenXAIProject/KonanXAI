from ..algorithm import Algorithm
from ....utils import *
from ....models import XAIModel
from ....datasets import Datasets
from ...core import darknet

import numpy as np
import cv2

class GradCAM(Algorithm):
    def __init__(self, model: XAIModel, dataset: Datasets, platform: PlatformType):
        super().__init__(model, dataset, platform)
        
    def calculate(self) -> list:
        self.result = []
        mtype = self.model.mtype
        # Input
        X = self.target_input
        # YOLO
        if mtype in (ModelType.Yolov4, ModelType.Yolov4Tiny, ModelType.Yolov5s):
            # Forward
            self.model.forward(X)
            # Search Bounding Box
            self._yolo_get_bbox()
            # Backward
            self._yolo_backward()
            # GradCAM
            heatmaps = self._gradcam()
            self.result.append(heatmaps)
            
        return self.result
    
    def _relu(self, x):
        return np.where(x > 0, x, 0)
    
    def _get_heatmap(self, feature, weight, size=(416, 416)):
        mul = feature * weight
        summation = np.sum(mul, axis=0)
        saliency_map = self._relu(summation)
        # Normalize
        smin, smax = saliency_map.min(), saliency_map.max()
        saliency = (saliency_map - smin) / (smax - smin)
        # Heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
        # Resize
        # TODO - Size 는 정해졌다고 가정, 임시
        resized = cv2.resize(heatmap, dsize=size, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def _gradcam(self):
        heatmaps = []
        for gradcam in self.gradcam:
            if isinstance(gradcam, list):
                # Multi Layer GradCAM
                maps = []
                for (feature, weight) in gradcam:
                    heatmap = self._get_heatmap(feature, weight)
                    maps.append(heatmap)
                heatmap = sum(maps)
            else:
                feature, weight = gradcam
                heatmap = self._get_heatmap(feature, weight)
            heatmaps.append(heatmap)
        return heatmaps
                
    # Platform Router
    def _yolo_get_bbox(self):
        if self.platform == PlatformType.Darknet:
            self._yolo_get_bbox_darknet()

    def _yolo_backward(self):
        if self.platform == PlatformType.Darknet:
            self._yolo_backward_darknet()
    
    # Darknet
    def _yolo_get_bbox_darknet(self):
        net: darknet.Network = self.model.net
        self.bboxes = []
        self.bbox_layer = {}
        for i, layer in enumerate(net.layers):
            if layer.type == darknet.LAYER_TYPE.YOLO:
                # TODO - Threadhold 관련은 config 통합 후 진행, 현재는 정적
                boxes = layer.get_bboxes(threshold=0.9)
                for box in boxes:
                    self.bbox_layer[box.entry] = i
                # Concat
                self.bboxes += boxes
        # TODO - NMS, 여기도 Threshold 정적
        if len(self.bboxes) > 1:
            self.bboxes = darknet.non_maximum_suppression_bboxes(self.bboxes, iou_threshold=0.5)

    def _yolo_backward_darknet(self):
        net: darknet.Network = self.model.net
        self.gradcam = []
        # TODO - Target Layer 는 정해졌다고 가정
        target_layer = [net.layers[29], net.layers[36]]
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
            for target in target_layer:
                feature = np.array(target.get_output())
                gradient = np.array(target.get_delta())
                stride = target.out_w * target.out_h
                # Reshape
                feature = feature.reshape((-1, stride))
                gradient = gradient.reshape((-1, stride)).mean(1)
                weight = gradient.reshape((-1, 1))
                # Append
                gradcam.append((feature, weight))
            self.gradcam.append(gradcam)


