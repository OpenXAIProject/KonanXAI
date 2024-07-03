from ...core.pytorch.yolov5s.utils import non_max_suppression, yolo_choice_layer
from ..algorithm import Algorithm
from ....utils import *
from ....models import XAIModel
from ....datasets import Datasets
from ...core import darknet
import torch
import numpy as np
import cv2

class GradCAM(Algorithm):
    def __init__(self, model: XAIModel, dataset: Datasets, platform: PlatformType):
        super().__init__(model, dataset, platform)       
    
    def set_model_hook(self):
        self.target_layer = self.model.target_layer
        if 'yolo' in self.model.mtype.name.lower():
            fwd_handle, bwd_handle = [],[]
            for layer in self.target_layer:
                layer.fwd_in = []
                layer.fwd_out = []
                layer.bwd_in = []
                layer.bwd_out = []
                fwd_handle.append(layer.register_forward_hook(self._hwd_hook))
                bwd_handle.append(layer.register_backward_hook(self._bwd_hook))
        else: 
            self.target_layer.fwd_in = []
            self.target_layer.fwd_out = []
            self.target_layer.bwd_in = []
            self.target_layer.bwd_out = []
            fwd_handle = self.target_layer.register_forward_hook(self._hwd_hook)
            bwd_handle = self.target_layer.register_backward_hook(self._bwd_hook)
        return fwd_handle, bwd_handle
    
    def _hwd_hook(self, l, fx, fy):
        l.fwd_in.append(fx[0])
        l.fwd_out.append(fy[0])
        
    def _bwd_hook(self, l, bx, by):
        print("ioninaisdnasd")
        l.bwd_in.insert(0, bx[0])
        l.bwd_out.insert(0, by[0])
    
    def calculate(self) -> list:
        self.result = None
        mtype = self.model.mtype
        # Input
        X = self.target_input
        if self.platform.name.lower() =='pytorch':
            X.requires_grad=True
            self.model.net.eval()
            self.model.net.requires_grad = True
            # Set target layer
            fwd_handle, bwd_handle = self.set_model_hook()
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
            self.result = heatmaps
            return self.result, self.bboxes
        else:
            # Forward
            self.model.forward(X)
            # Backward
            self._torch_backward()
            # Gradcam
            heatmaps = self._gradcam()
            self.result = heatmaps
            #result
            fwd_handle.remove()
            bwd_handle.remove()
            return self.result
    
    def _relu(self, x):
        return np.where(x > 0, x, 0)
    
    def _normalized(self, saliency_map):
        smin, smax = saliency_map.min(), saliency_map.max()
        if (smax - smin != 0):
            saliency = (saliency_map - smin) / (smax - smin)
        else:
            saliency = saliency_map
            
        return saliency
    
    def _get_heatmap(self, feature, weight):#size=(640, 640)):
        mul = feature * weight
        summation = np.sum(mul, axis=0)
        heatmap = self._relu(summation)
        resized = cv2.resize(heatmap, dsize=self.dataset.fit, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def _norm_heatmap(self, maps):
        heatmap = sum(maps)
        smin, smax = heatmap.min(), heatmap.max()
        if (smax - smin != 0):
            saliency = (heatmap - smin) / (smax - smin)
        else:
            saliency = heatmap
        return saliency
    
    def _gradcam(self):
        heatmaps = []
        for gradcam in self.gradcam:
            if isinstance(gradcam, list):
                # Multi Layer GradCAM
                maps = []
                for (feature, weight) in gradcam:
                    heatmap = self._get_heatmap(feature, weight)
                    maps.append(heatmap)
                saliency = self._norm_heatmap(maps)
                # Heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
            else:
                feature, weight = gradcam
                heatmap = self._get_heatmap(feature, weight)
                heatmap = self._norm_heatmap([heatmap])
                heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmaps.append(heatmap)
        return heatmaps
                
    # Platform Router
    def _yolo_get_bbox(self):
        if self.platform == PlatformType.Darknet:
            self._yolo_get_bbox_darknet()
        elif self.platform == PlatformType.Pytorch:
            self._yolo_get_bbox_pytorch()
           

    def _yolo_backward(self):
        if self.platform == PlatformType.Darknet:
            self._yolo_backward_darknet()
        elif self.platform == PlatformType.Pytorch:
            self._yolo_backward_pytorch()
    # Darknet
    def _yolo_get_bbox_darknet(self):
        net: darknet.Network = self.model.net
        self.bboxes = []
        self.bbox_layer = {}
        for i, layer in enumerate(net.layers):
            if layer.type == darknet.LAYER_TYPE.YOLO:
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
        net: darknet.Network = self.model.net
        self.gradcam = []
        # TODO - Target Layer 는 정해졌다고 가정
        # target_layer = [net.layers[149], net.layers[149], net.layers[160]]
        # target_layer = [net.layers[30], net.layers[37]]
        select_layer = set(list(self.bbox_layer.values()))
        print(f"select_layer: {select_layer}")
        for box in self.bboxes:
            i = self.bbox_layer[box.entry]
            layer = net.layers[i -1]
            out = layer.get_output()
            net.zero_grad()
            # feature index
            stride = layer.out_w * layer.out_h
            idx = box.entry + (5 + box.class_idx) * stride
            # set delta
            layer.delta[idx] = out[idx]
            net.backward()
            # Get Features
            target = layer
            # for target in target_layer:
            feature = np.array(target.get_output())
            gradient = np.array(target.get_delta())
            stride = target.out_w * target.out_h
            # Reshape
            feature = feature.reshape((-1, target.out_w, target.out_h))
            gradient = gradient.reshape((-1, stride)).mean(1)
            weight = gradient.reshape((-1, 1, 1))
            # Append
            self.gradcam.append((feature, weight))
        # Pytorch
    def _torch_backward(self):
        self.gradcam = []
        preds = self.model.last_outputs
        label_index = torch.argmax(preds).item()
        #backward hook 확인.
        preds[0][label_index].backward()
        feature = self.target_layer.fwd_in[-1].detach().cpu().numpy()
        gradient = self.target_layer.bwd_in[-1].detach().cpu().numpy()
        
        feature = feature.reshape((-1, feature.shape[-1], feature.shape[-2]))
        gradient = gradient.reshape((-1, gradient.shape[-1], gradient.shape[-2]))
        
        self.gradcam.append((feature, gradient))
        
    def _yolo_get_bbox_pytorch(self):
        self.preds_origin, raw_logit = self.model.last_outputs
        self.logits_origin = torch.concat([data.view(-1,self.preds_origin.shape[-1])[...,5:] for data in raw_logit],dim=0)
        with torch.no_grad():
            self.preds, logits, self.select_layers = non_max_suppression(self.preds_origin, self.logits_origin.unsqueeze(0), conf_thres=0.25, model_name = self.model.mtype.name)
        self.index_tmep = yolo_choice_layer(raw_logit, self.select_layers)
        
    def _yolo_backward_pytorch(self):
        saliency_maps = []
        class_index = []
        for cls, sel_layer in zip(self.preds[0], self.select_layers):
            self.model.net.zero_grad()
            self.logits_origin[sel_layer][int(cls[5].item())].backward()
            box_saliency_maps = []
            