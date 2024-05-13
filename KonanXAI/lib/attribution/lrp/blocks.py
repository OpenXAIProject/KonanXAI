import yaml
import torch
import torch.nn as nn
import numpy as np
from ....models import XAIModel

# Main
class Blocks:
    def __init__(self, *args, **kwargs):
        self.weight = None
        self.input = None

    def relevance(self, rel, alpha=None, beta=None, eps=None):
        pass

# Operator Blocks
class Add(Blocks):
    pass

class Clone(Blocks):
    pass

class Concat(Blocks):
    pass

# Base Blocks
class Linear(Blocks):
    pass

class Conv2d(Blocks):
    pass

class BatchNorm(Blocks):
    pass

class MaxPool2d(Blocks):
    pass

class ReLU(Blocks):
    pass

class SiLU(Blocks):
    pass

class Sigmoid(Blocks):
    pass

# Yolo Blocks
class Conv(Blocks):
    pass

class Upsample(Blocks):
    pass

class BottleNeck(Blocks):
    pass

class CBS(Blocks):
    # Conv + BN + SiLU
    pass

class C3(Blocks):
    # CBS*3 + BottleNeck
    pass

class SPPF(Blocks):
    # CBS*2 + MaxPool*3
    pass

class Detect(Blocks):
    pass

# Blocks
class Graph:
    # TODO: Relevance
    def __init__(self, model: XAIModel, conf_path: str):
        self.layers: list[Blocks] = []
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
        
        print("INDEX\t\tINPUT\t\tModule\t\tArgs")
        # In-Index, Repeat, Module, args
        for i, (f, n, m, args) in enumerate(backbone + head):
            m = m.replace("nn.", "")
            module = eval(m)
            # Convert Args
            for j, arg in enumerate(args):
                try:
                    args[j] = eval(arg) if isinstance(arg, str) else arg
                except:
                    pass
            n = max(round(n * gd), 1) if n > 1 else n
            idx = i+f if f == -1 else f
            print(f"[{i}]\t\t{idx}\t\t{module.__name__}\t\t{args}")

    def relevance(self, rel, alpha=None, beta=None, eps=None):
        pass