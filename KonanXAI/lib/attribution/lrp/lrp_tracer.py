import yaml
import numpy as np

from torch.fx.node import Node
from typing import Dict

from ...attribution.lrp.lrp_rule import Input, Clone, Add, Sequential, Flatten
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

class Graph:
    # TODO: Relevance
    def __init__(self, model: XAIModel, conf_path: str = None):
        self.model = model
        self.input = Input()
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
    
    def trace(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.

        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.

        Returns:
            Layers of original model stored in module list.

        """
        gt_graph = symbolic_trace(self.model)
        trace_graph = []
        residual_layer = []
        self.index = 0
        call_func_layer = []
        self.variable = {}
        self.modules = dict(self.model.named_modules())

        for v in gt_graph.nodes:
            if v.op == 'placeholder':
                trace_graph.append(self.input)
            elif v.op == 'call_module':
                target = self._target_module(v)
                if str(v.prev).startswith(("mul", "add")):
                    residual_layer.append(target)
                    trace_graph.append(Sequential(residual_layer))
                    residual_layer.clear()
                else:
                    if len(self.variable)>0:
                        if v.name in str(self.variable.keys()):
                            self.x0 = Sequential(call_func_layer)
                            call_func_layer.clear()
                        call_func_layer.append(target)
                    else:
                        trace_graph.append(target)
            elif v.op == 'call_function':
                if v.name.startswith("add"):
                    if str(list(self.variable.keys())[-1]).startswith("add"):
                        self.x0 = Sequential(call_func_layer)
                        self.x1 = Sequential([self.input])
                    else:
                        m = Sequential(call_func_layer)
                        self.x1 = Sequential([self.input, m])
                    call_func_layer.clear()           
                    add = Add(self.x0, self.x1)
                    residual_layer.append(add)
                    self.variable.clear()
                elif v.name.startswith("mul"):
                    pass
                elif v.name.startswith("flatten"):
                    trace_graph.append(Flatten(self.modules[str(v.prev)]))
                    
                # print("function name: ",v.name, "argument: ",v.args, "usage: ",v.users)
            if len(v.users)>1:
                # target = self._target_module(v)
                def input_forward_hook(m, input_tensor):
                    hook_idx = [v.id for v in self.input.handle]
                    key = list(m._forward_pre_hooks.keys())[-1]
                    if key in hook_idx:
                        self.input.X[key]= input_tensor[0]
                self.input.handle.append(self._target_module(v.next).register_forward_pre_hook(input_forward_hook))
                clone = Clone(self.modules[str(v.target)], num=len(v.users))
                next_name = list(v.users.keys())[-1]
                self.variable[next_name] = v.op
                residual_layer.append(clone)
        # print(trace_graph)

        return Sequential(trace_graph)