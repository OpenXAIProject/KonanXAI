import yaml
import numpy as np

from torch.fx.node import Node
from typing import Dict

from ...attribution.lrp.lrp_rule import Input, Clone, Add, Sequential, Flatten, Mul, StochasticDepth
from ....models import XAIModel
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph_module import GraphModule
def symbolic_trace(module):
    tracer = Tracer()
    tracer.traced_func_name = module.forward.__name__
    graph = tracer.trace(module)
    name = module.__class__.__name__ if isinstance(module, torch.nn.Module) else module.__name__
    graph = GraphModule(tracer.root, graph, name).graph
    return graph
class SplitFunc:
    def __init__(self):
        self.destination = None
        self.next = False
        self.X0 = []
        self.X1 = []
    def _clear(self):
        self.destination = None
        self.X0.clear()
        self.X1.clear()
class Graph:
    # TODO: Relevance
    def __init__(self, model: XAIModel, conf_path: str = None):
        self.model = model
        self.input = Input()
        self._layer_name = None
        if conf_path is not None:
            # Yolo
            self._load_conf(conf_path)
        
    def _load_conf(self, conf_path:str):
        with open(conf_path, encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.data = data
        self._parse_graph_yaml()
        
    def _parse_graph_yaml(self):
        nc = self.data['nc']
        gd = self.data['depth_multiple']
        gw = self.data['width_multiple']
        anchors = self.data['anchors']
        backbone = self.data['backbone']
        head = self.data['head']
        
    def _target_module(self, v):
        target = v.target.split(".")
        target_module = self.model
        for value in target:
            if value.isdecimal():
                target_module = target_module[int(value)]
            else:
                target_module = getattr(target_module,value)
        return target_module
    
    def _next_layer_name(self, v):
        layer_names = v.target.split(".")
        if len(layer_names) == 1:
            return None
        else:
            if layer_names[0][-1].isdigit():
                name = layer_names[0]
            else:
                name = layer_names[0] + layer_names[1]
            return name
        
    def _target_layer_name(self, v):
        layer_names = v.target.split(".")
        prev_layer = self._layer_name
        if len(layer_names) == 1:
            self._layer_name = None
            return False
        else:
            if layer_names[0][-1].isdigit():
                self._layer_name = layer_names[0]
            else:
                self._layer_name = layer_names[0] + layer_names[1]
            cur_layer = self._layer_name
            
        if prev_layer == cur_layer:
            return False
        else:
            return True
                
    def trace(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.

        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.

        Returns:
            Layers of original model stored in module list.

        """
        gt_graph = symbolic_trace(self.model)
        trace_graph = []
        main_layer = []
        sub_layer = []
        call_func_layer = []
        self.variable = {}
        self.dict_name = []
        self.stocastic = {}
        self.add = {}
        self.modules = dict(self.model.named_modules())
        for v in gt_graph.nodes:
            if v.op == 'placeholder':
                trace_graph.append(self.input)
            elif v.op == 'call_module':
                is_next_layer = self._target_layer_name(v)
                target = self._target_module(v)
                if is_next_layer == True:
                    if len(main_layer)>0:
                        if len(call_func_layer)>0:
                            sub_layer.append(Sequential(call_func_layer))
                            call_func_layer.clear()
                        if len(sub_layer)>0 and not isinstance(sub_layer[0],Clone):
                            main_layer.append(Sequential(sub_layer))
                            sub_layer.clear()
                        trace_graph.append(Sequential(main_layer))
                        main_layer.clear()
                    if len(self.variable)>0:
                        if not v.name in self.variable[self.dict_name[-1]].destination:
                            self.variable[self.dict_name[-1]].X0.append(target)
                        else:
                            self.variable[self.dict_name[-1]].X1.append(target)
                    else:
                        if len(call_func_layer)>0 and len(sub_layer)==0:
                            trace_graph.append(Sequential(call_func_layer))
                        else:
                            if len(call_func_layer)==0:
                                pass
                            else:   
                                sub_layer.append(Sequential(call_func_layer))
                        call_func_layer.clear()
                        call_func_layer.append(target)
                    
                elif is_next_layer == False and self._layer_name == None:
                    if len(main_layer)>0:
                        trace_graph.append(Sequential(main_layer))
                        main_layer.clear()
                    elif len(call_func_layer)>0:
                        trace_graph.append(Sequential(call_func_layer))
                        call_func_layer.clear()
                    trace_graph.append(target)
                elif len(self.variable)>0 :
                    if not v.name in self.variable[self.dict_name[-1]].destination or self.variable[self.dict_name[-1]].next == False:
                        self.variable[self.dict_name[-1]].X0.append(target)
                    else:
                        self.variable[self.dict_name[-1]].X1.append(target)
                        self.variable[self.dict_name[-1]].destination = v.next.name
                elif len(v.users)>1 and len(sub_layer)>0:
                    call_func_layer.append(target)
                    sub_layer.append(Sequential(call_func_layer))
                    main_layer.append(Sequential(sub_layer))
                    sub_layer.clear()
                    call_func_layer.clear()
                elif self._next_layer_name(v.next) == None:
                    if len(call_func_layer)>0:
                        call_func_layer.append(target)
                    else:
                        sub_layer.append(target)
                    if len(sub_layer)>0:
                        if len(call_func_layer)>0:
                            sub_layer.append(Sequential(call_func_layer))
                        call_func_layer.clear()
                    main_layer.append(Sequential(sub_layer))
                    sub_layer.clear()
                else:
                    call_func_layer.append(target)
            elif v.op == 'call_function':
                if v.name.startswith("add"):
                    dict_name = self.dict_name.pop()
                    multi_clone = []
                    del_count = 0
                    if len(sub_layer)>1:
                        for index, value in enumerate(sub_layer):
                            if index == 0:
                                continue
                            else:
                                multi_clone.append(value)
                                del_count+=1
                        del sub_layer[1:1+del_count]
                    x0 = Sequential(self.variable[dict_name].X0)
                    if len(self.variable[dict_name].X1)>0:
                        x1 = Sequential([self.input,Sequential(self.variable[dict_name].X1)])
                    else:
                        x1 = Sequential([self.input])
                    if len(multi_clone)>0:
                        multi_clone.append(x0)
                        x0 = Sequential(multi_clone)
                        multi_clone.clear()
                    add = Add(x0, x1)
                    self.add[v.name] = add
                    sub_layer.append(add)
                    del self.variable[dict_name]
                elif v.name.startswith("mul"):
                    dict_name = self.dict_name.pop()
                    x0 = Sequential(self.variable[dict_name].X0)
                    if len(self.variable[dict_name].X1)>0:
                        x1 = Sequential([self.input,Sequential(self.variable[dict_name].X1)])
                    else:
                        x1 = Sequential([self.input])
                    mul = Mul(x0, x1)
                    sub_layer.append(mul)
                    del self.variable[dict_name]
                elif v.name.startswith("flatten"):
                    trace_graph.append(Flatten(self.modules[str(v.prev)]))
                elif v.name.startswith("stochastic_depth"):
                    layer, p, mod, training = v.args
                    _module = StochasticDepth(self.modules[str(layer.target)],p,mod,training)
                    # _module = StochasticDepth(v.target)
                    self.stocastic[str(v.name)] = _module
                    if not v.name in self.variable[self.dict_name[-1]].destination or self.variable[self.dict_name[-1]].next == False:
                        self.variable[self.dict_name[-1]].X0.append(_module)
                    else:
                        self.variable[self.dict_name[-1]].X1.append(_module)
                        self.variable[self.dict_name[-1]].destination = v.next.name
                    # call_func_layer가 아닌 dict 내부의 연산값에 저장해야함
                    # call_func_layer.append(_module)
            # clone
            if len(v.users)>1:
                # target = self._target_module(v)
                def input_forward_hook(m, input_tensor):
                    hook_idx = [v.id for v in self.input.handle]
                    key = list(m._forward_pre_hooks.keys())[-1]
                    if key in hook_idx:
                        self.input.X[key]= input_tensor[0]
                self.input.handle.append(self._target_module(v.next).register_forward_pre_hook(input_forward_hook))
                try:
                    clone = Clone(self.modules[str(v.target)], num=len(v.users))
                except:
                    # 다른 방법 처리 필요.. stocastic_dept를 어떻게 가져올까...
                    try:
                        module = (self.stocastic[str(v.args[0])] , self.modules[v.args[1].target])
                    except:
                        module = (self.stocastic[str(v.args[0])] , self.add[str(v.args[1])])
                    clone = Clone(module, num = len(v.users))
                next_name = list(v.users.keys())[-1]
                self.dict_name.append(str(v.name))
                self.variable[v.name] = SplitFunc()
                if str(next_name).startswith(("add", "mul")):
                    self.variable[v.name].destination = str(next_name.prev)
                    self.variable[v.name].next = False
                else:
                    self.variable[v.name].destination = str(next_name)
                    self.variable[v.name].next = True
                if len(self.variable)>1 and len(self.variable[self.dict_name[0]].X0)>0:
                    sub_layer.append(Sequential(self.variable[self.dict_name[0]].X0))
                    self.variable[self.dict_name[0]].X0.clear()
                    sub_layer.append(clone)
                    pass
                else:
                    if len(call_func_layer)>0:
                        sub_layer.append(Sequential(call_func_layer))
                        call_func_layer.clear()
                    if len(sub_layer)>0:
                        main_layer.append(Sequential(sub_layer))
                        sub_layer.clear()
                    sub_layer.append(clone)
                
        if len(call_func_layer)>0:
            trace_graph.append(Sequential(call_func_layer))

            
        return Sequential(trace_graph) 