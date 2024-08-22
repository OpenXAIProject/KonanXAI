import yaml
import numpy as np
import copy
from torch.fx.node import Node
from typing import Dict

from ...attribution.layer_wise_propagation.lrp_rule import Input, Clone, Add, Sequential, Flatten, Mul, StochasticDepth, Cat, Detect, Upsample, Route
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
    def __init__(self, model, conf_path: str = None):
        self.model = model
        self.input = Input()
        self.cat = Cat()
        self.conf_path = conf_path
        self._layer_name = None
        
    def _load_conf(self, conf_path:str):
        with open(conf_path, encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.data = data
        architecture = self._parse_graph_yaml()
        return architecture
    def clear(self):
        self.input.clear()
        self.cat.clear()
    def _parse_graph_yaml(self):
        nc = self.data['nc']
        gd = self.data['depth_multiple']
        gw = self.data['width_multiple']
        anchors = self.data['anchors']
        backbone = self.data['backbone']
        head = self.data['head']
        return backbone+head
       
    def _target_module(self, v):
        if self.conf_path != None:
            target = v.target
            target_module = self.modules[self.tree_index][target]
            return target_module
        else:
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
        if self.conf_path == None:
            gt_graph = symbolic_trace(self.model).nodes
            self.modules = dict(self.model.named_modules())
            return self.make_trace(gt_graph)
        else:
            yaml_layer = self._load_conf(self.conf_path)
            self.modules = {}
            gt_graph = {}
            for index, m in enumerate(self.model):
                self.tree_index = index
                temp_layer = []
                if m._get_name() == 'Concat':
                    layer_index = yaml_layer[index][0]
                    cur_index = index -1
                    next_index = layer_index[1]
                    gt_graph[index] = Route(cur_index, next_index)
                    # print("Concat layer!", yaml_layer[index])
                    continue
                elif m._get_name() == 'Detect':
                    detect_layers = yaml_layer[index][0]
                    layer = [m.m._modules[str(n)] for n in range(len(detect_layers))]
                    gt_graph[index] = Detect(detect_layers, layer)
                    # print("Detect Layer!", yaml_layer[index])
                    continue
                else:
                    self.modules[index] = dict(m.named_modules())
                for v in symbolic_trace(m).nodes: 
                    if index !=0 and v.name=='x':
                        continue
                    elif v.name == "output":
                        continue
                    else:
                        temp_layer.append(v)
                gt_graph[index] = self.make_trace(temp_layer)
            return Sequential(list(gt_graph.values()))
        
    def make_trace(self, gt_graph):
        trace_graph = []
        main_layer = []
        sub_layer = []
        call_func_layer = []
        self.concat = {}
        self.variable = {}
        self.dict_name = []
        self.stocastic = {}
        self.add = {}
        bottle_neck = []
        
        for v in gt_graph:
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
                        if len(sub_layer)>1:
                            main_layer.append(Sequential(sub_layer))
                            sub_layer.clear()
                        trace_graph.append(Sequential(main_layer))
                        main_layer.clear()
                    if len(self.variable)>0:
                        if not v.name in self.variable[self.dict_name[-1]].destination or self.variable[self.dict_name[-1]].next == False:
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
                    if len(main_layer)>0 and len(bottle_neck)==0:
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
                    elif len(sub_layer)==0 and len(call_func_layer)>0:
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
                    add = Add(x0, x1)
                    x0 = None
                    x1 = None
                    self.add[v.name] = add
                    
                    sub_layer.append(add)
                    if len(multi_clone)>0:
                        sub_layer.extend(multi_clone)
                        multi_clone.clear()
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
                elif v.name.startswith("cat"):
                    clone = Clone(self.modules[self.tree_index][str(gt_graph[0].target)], num=len(self.concat))
                    trace_graph.append(clone)
                    cat = Cat(*bottle_neck)
                    trace_graph.append(cat)
                    if len(main_layer)>0:
                        trace_graph.insert(0, main_layer[0][0])
                        main_layer.clear()
                    self.concat.clear()
                elif v.name.startswith("interpolate"):
                    trace_graph.pop()
                    upsample_param = {}
                    for key,value in v._kwargs.items():
                        upsample_param[key] = value
                    del upsample_param['antialias']
                    trace_graph.append(Upsample(upsample_param))
                elif v.name.startswith("flatten"):
                    trace_graph.append(Flatten(self.modules[str(v.prev)]))
                elif v.name.startswith("stochastic_depth"):
                    layer, p, mod, training = v.args
                    _module = StochasticDepth(self.modules[str(layer.target)],p,mod,training)
                    self.stocastic[str(v.name)] = _module
                    
                    if not v.name in self.variable[self.dict_name[-1]].destination or self.variable[self.dict_name[-1]].next == False:
                        sub_layer.append(Sequential(self.variable[self.dict_name[-1]].X0))
                        self.variable[self.dict_name[-1]].X0.clear()
                        self.variable[self.dict_name[-1]].X0.append(_module)
                    else:
                        self.variable[self.dict_name[-1]].X1.append(_module)
                        self.variable[self.dict_name[-1]].destination = v.next.name
            # Cat
            if "cat" in list(map(str,v.users.keys())):
                if v.name.startswith("add"):
                    self.cat.handle.append(self._target_module(v.next))
                    self.concat[v.name] = self.modules[self.tree_index][str(v.next.target)]
                elif("act" in v.name):
                    self.cat.handle.append([self._target_module(v).layer_count, self._target_module(v)])
                    self.concat[v.name] = self.modules[self.tree_index][str(v.target)]
                else:
                    self.cat.handle.append(self._target_module(v))
                    self.concat[v.name] = self.modules[self.tree_index][str(v.target)]
                if len(call_func_layer)>0 and len(sub_layer)==0:
                    trace_graph.append(Sequential(call_func_layer))
                    call_func_layer.clear()
                elif len(sub_layer)>0:
                    if len(sub_layer) == 1:
                        trace_graph.append(sub_layer[0])
                    else:
                        trace_graph.append(Sequential(sub_layer))
                    sub_layer.clear()
                elif len(main_layer)>0 and len(trace_graph)==0:
                    module = main_layer[0][0].modules.pop()
                    trace_graph.append(module)

                bottle_neck.append(Sequential(trace_graph))
                trace_graph.clear()
            # clone
            if len(v.users)>1 and not "cat" in list(map(str,v.users.keys())):
                def input_forward_hook(m, input_tensor):
                    hook_idx = [v.id for v in self.input.handle]
                    key = list(m._forward_pre_hooks.keys())[-1]
                    if key in hook_idx:
                        self.input.X[key]= input_tensor[0]
                self.input.handle.append(self._target_module(v.next).register_forward_pre_hook(input_forward_hook))
                try:
                    if self.conf_path != None:
                        clone = Clone(self.modules[self.tree_index][str(v.target)],idx = self.modules[self.tree_index][str(v.target)].layer_count, num=len(v.users))
                    else:
                        clone = Clone(self.modules[str(v.target)], num=len(v.users))
                except:
                    if self.conf_path != None:
                        clone = Clone(self.modules[self.tree_index][str(v.next.target)],num=len(v.users))
                    else:
                        clone = Clone(self.modules[str(v.next.target)],num=len(v.users))
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
                
        if len(sub_layer)>0 or len(call_func_layer)>0 or len(main_layer)>0:
            if len(call_func_layer)>0:
                sub_layer.append(Sequential(call_func_layer))
                call_func_layer.clear()
            if len(sub_layer)>0:
                if isinstance(sub_layer[0],Sequential) and len(sub_layer) == 1:
                    main_layer.append(sub_layer[0])
                else:
                    main_layer.append(Sequential(sub_layer))
                sub_layer.clear()
            if len(main_layer)>0:
                if isinstance(main_layer[0],Sequential) and len(main_layer) == 1:
                    trace_graph.append(main_layer[0])
                else:
                    trace_graph.append(Sequential(main_layer))
                main_layer.clear()
        return Sequential(trace_graph)