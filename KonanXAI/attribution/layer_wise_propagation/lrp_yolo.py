from KonanXAI.attribution.layer_wise_propagation.lrp_tracer import Graph
from ..._core.pytorch.yolov5s.utils import non_max_suppression, yolo_choice_layer
from ...utils import *

from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import copy
class LRPYolo: 
    def __init__(self, 
            framework, 
            model, 
            input, 
            config):
        self.framework = framework
        self.model = model
        self.model_name = self.model.model_name
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input = input[0].to(device)
        
        self.rule = config['rule']
        self.alpha = None
        self.yaml_path = config['yaml_path']
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
        model =  copy.deepcopy(self.model.fuse())
        model.model.eval()
        self.tracer =  Graph(model.model, self.yaml_path)
        self.module_tree = self.tracer.trace()
        
        #bottle neck hook>??
        self.preds_origin, raw_logit = model(self.input)
        
        self.logits_origin = torch.concat([data.view(-1,self.preds_origin.shape[-1])[...,5:] for data in raw_logit],dim=0)
        with torch.no_grad():
            self.preds, logits, self.select_layers = non_max_suppression(self.preds_origin, self.logits_origin.unsqueeze(0), conf_thres=0.25, model_name = self.model.model_name)
        self.index_tmep = yolo_choice_layer(raw_logit, self.select_layers)
        
        tensor_shape = [tuple(x.shape) for x in raw_logit] #1,3,80,80,85
        new_tensor = self.tensor_parsing(self.preds_origin[0], tensor_shape)
        
        relevance_li = []
        bbox_li = []
        for cls, sel_layer, sel_layer_index in zip(self.preds[0], self.select_layers, self.index_tmep):
            cls_index = int(cls[5].item())
            cat_layer = {}
            relevance = self.detect_relevance(new_tensor, sel_layer_index,rel_for_class=cls_index)
            #sort
            skip_count = 0
            flag = 0
            bbox_li.append(cls[:4])
            for indexs, module in enumerate(reversed(self.module_tree.modules)):
                index = len(self.module_tree.modules) -1 - indexs
                # print(index)
                if flag and isinstance(relevance, dict) and flag==1:
                    layer_index = self.tracer.data['head'][-1][0]
                    del_relevance = []
                    for v in layer_index:
                        del_relevance.append(layer_index[-1]-v)
                    skip_count = layer_index[-1] - layer_index[sel_layer_index]
                    if skip_count>0:
                        key = str(layer_index[del_relevance.index(skip_count)])
                        rel = cat_layer[key]
                        relevance.clear()
                        relevance[key] = rel
                    flag = False
                if skip_count>0:
                    skip_count -= 1
                    # print("skip  ",index)
                    continue
                if str(index) in cat_layer:
                    if isinstance(relevance,dict):
                        key, rel = next(iter(relevance.items()))
                        if key != str(index):
                            relevance[str(index)] = rel + cat_layer[str(index)]
                            del relevance[key]
                    else:
                        relevance = relevance + cat_layer[str(index)]
                    del cat_layer[str(index)]
                relevance = module.backprop(relevance, self.rule, self.alpha)
                flag+=1
                if len(relevance)>1:
                    for index,(key, rel) in enumerate(relevance.items()):
                        if index == 0:
                            continue
                        else:
                            if key == 'upsample':
                                del_flag = False
                                continue
                            cat_layer[key] = rel
                            delete_key = key
                            del_flag = True
                    if del_flag != False:
                        del relevance[delete_key]            

            relevance = relevance.sum(dim=1, keepdim=True)
            relevance_li.append(relevance.detach().cpu())
            print("complete")
            self.module_tree.clear()
        return relevance_li, bbox_li
    
    def detect_relevance(self, pred, detect_layer, rel_for_class):
        self.box = None
        self.conf = False
        self.max_class_only = False
        self.contrastive = False
        self.rel_for_class = rel_for_class
        nc = pred[0].shape[-1] - 5
        initial_relevance = {}
        norm = 0
        prop_to = self.model.yaml['head'][-1][0]
        for to, x in zip(prop_to, pred):
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
            initial_relevance[str(to)] = x
        return initial_relevance