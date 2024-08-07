from KonanXAI.lib.attribution.lrp.lrp_tracer import Graph
from ...core.pytorch.yolov5s.utils import non_max_suppression, yolo_choice_layer
from ....models import XAIModel
from ....datasets import Datasets
from ..algorithm import Algorithm
from ....utils import *

## test
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import copy
class LRPYolo(Algorithm): 
    def __init__(self, model: XAIModel, dataset: Datasets, platform: PlatformType):
        super().__init__(model, dataset, platform)
        self.rule = self.model.rule
        self.alpha = None
        self.yaml_path = self.model.yaml_path
        if self.alpha != 'None':
            try:
                self.alpha = int(self.alpha)
            except: #epsilon rule default = 1e-8
                self.alpha = sys.float_info.epsilon
                
    def tensor_parsing(self,pred, t_shape):
        tensor_1 = pred[:t_shape[0][2]*t_shape[0][3]*t_shape[0][1]].view(t_shape[0])
        if len(t_shape)==3:
            tensor_2 = pred[t_shape[0][2]*t_shape[0][3]*t_shape[0][1]:t_shape[0][2]*t_shape[0][3]*t_shape[0][1]+t_shape[1][2]*t_shape[1][3]*t_shape[1][1]].view(t_shape[1])
            tensor_3 = pred[t_shape[0][2]*t_shape[0][3]*t_shape[0][1]+t_shape[1][2]*t_shape[1][3]*t_shape[1][1]:].view(t_shape[2])
            return [tensor_1, tensor_2, tensor_3]
        elif len(t_shape)==2:
            tensor_2 = pred[t_shape[0][2]*t_shape[0][3]*t_shape[0][1]:].view(t_shape[1])
            return[tensor_1, tensor_2]
        
    def calculate(self):
        model =  copy.deepcopy(self.model.net.fuse())
        model.model.eval()
        self.tracer =  Graph(model.model, self.yaml_path)
        # tracer 수정 중
        self.module_tree = self.tracer.trace()
        
        
        self.preds_origin, raw_logit = model(self.target_input)
        self.logits_origin = torch.concat([data.view(-1,self.preds_origin.shape[-1])[...,5:] for data in raw_logit],dim=0)
        with torch.no_grad():
            self.preds, logits, self.select_layers = non_max_suppression(self.preds_origin, self.logits_origin.unsqueeze(0), conf_thres=0.25, model_name = self.model.mtype.name)
        self.index_tmep = yolo_choice_layer(raw_logit, self.select_layers)
        
        tensor_shape = [tuple(x.shape) for x in raw_logit] #1,3,80,80,85
        new_tensor = self.tensor_parsing(self.preds_origin[0], tensor_shape)
        
        relevance_li = []
        for cls, sel_layer, sel_layer_index in zip(self.preds[0], self.select_layers, self.index_tmep):
            relevance = self.detect_relevance(new_tensor, sel_layer_index)
            relevance = self.module_tree.backprop(relevance, self.rule, self.alpha)
            
            relevance = torch.sum(relevance, axis=1)
            relevance = relevance.unsqueeze(1)
            relevance_li.append(relevance.cpu().detach())
        return relevance
    
    def detect_relevance(self, pred, detect_layer):
        self.box = None
        self.conf = False
        self.max_class_only = False
        self.contrastive = False
        self.rel_for_class = 0
        nc = pred[0].shape[-1] - 5
        initial_relevance = {}
        norm = 0
        yaml_detec_layer = self.model.net.yaml['head'][-1][0]
        x = pred[detect_layer]
        
        xs = x.size()
        x = x.clone()

        x = x.view(-1, nc+5)            
        if self.box is not None :
            i = ((x[:, 0] < self.box[0]) + (x[:, 1] < self.box[1]) +
                    (x[:, 0] > self.box[2]) + (x[:, 1] > self.box[3]))
            x[i, 4:] = torch.zeros_like(x[i, 4:])
        if self.conf :
            x[:, 5:] = x[:, [4]] * x[:, 5:]
        x[:, :5] = torch.zeros_like(x[:, :5])
        max_class, i = x[:, 5:].max(dim=1, keepdim=True)
        if self.max_class_only :
            x[:, 5:] = torch.zeros_like(x[:, 5:]).scatter(-1, i, max_class)
        if self.rel_for_class is not None :
            if self.contrastive :
                dual = x.clone()
                dual[:, self.rel_for_class] = torch.zeros(dual.size(0))
                max_class_dual, i_dual = dual[:, 5:].max(dim=1, keepdim=True)
                dual[:,5:] = torch.zeros_like(dual[:, 5:]).scatter(-1, i_dual, max_class_dual)
            x[:, 5:5+self.rel_for_class] = torch.zeros_like(x[:, 5:5+self.rel_for_class])
            x[:, self.rel_for_class+6:] = torch.zeros_like(x[:, self.rel_for_class+6:])
        nonzero= torch.nonzero(x)
        if self.contrastive :
            x = torch.cat([x.view(xs), dual.view(xs)], dim=0)
        else :
            x = x.view(xs)
        norm += x.sum()
        initial_relevance[detect_layer] = x
        return initial_relevance