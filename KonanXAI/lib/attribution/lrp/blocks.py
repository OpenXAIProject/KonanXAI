import yaml
import torch
import torch.nn as nn
import numpy as np
from ....models import XAIModel

# Main
class Blocks:
    def __init__(self, *args, **kwargs):
        self.layers = []
        self.weight = None
        self.input = None
        
    def add_layers(self, layer: object):
        self.layers.append(layer)
        
    def __repr__(self):
        out = ""
        if len(self.layers) > 0:
            for layer in self.layers:
                out += str(layer) + "\n"
        else:
            out = self.__class__.__name__
        return out

    def relevance(self, rel, alpha=None, beta=None, eps=None):
        pass

# Operator Blocks
class Add(Blocks):
    pass

class Clone(Blocks):
    pass
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     for layer in args:
    #         self.add_layers(layer)

class Concat(Blocks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in args:
            self.add_layers(layer)

class Sequential(Blocks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in args:
            self.add_layers(layer)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_layers(Conv2d())
        self.add_layers(BatchNorm())
        self.add_layers(SiLU())

class Upsample(Blocks):
    pass

class BottleNeck(Blocks):
    def __init__(self, *args, shortcut=True, **kwargs):
        super().__init__(*args, **kwargs)
        seq = Sequential(*[Conv(), Conv()])
        add = None
        if shortcut and kwargs['c1'] == kwargs['c2']:
            clone = Clone(self, 2)
            self.add_layers(clone)
            add = Add(clone, seq)
        self.add_layers(seq)
        if add is not None:
            self.add_layers(add)

class C3(Blocks):
    # CBS*3 + BottleNeck
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cv1 = Conv()
        cv2 = Conv()
        cv3 = Conv()
        shortcut = True if kwargs['c1'] is None else kwargs['c1']
        seq = [BottleNeck(*args, shortcut=shortcut, **kwargs) for _ in range(kwargs['n'])]
        seq = [cv1] + seq
        seq = Sequential(*seq)
        cat = Concat(seq, cv2)
        self.add_layers(cat)
        self.add_layers(cv3)

class SPPF(Blocks):
    # CBS*2 + MaxPool*3
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cv1 = Conv()
        cv2 = Conv()
        m = MaxPool2d()
        # TODO: 고민 필요 부분
        y1 = Sequential(*[cv1, m])
        y2 = Sequential(*[y1, m])
        y3 = Sequential(*[y2, m])
        cat = Concat(cv1, y1, y2, y3)
        self.add_layers(cat)
        self.add_layers(cv2)

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
            c0, c1, c2, c3 = [None, None, None, None]
            for j, arg in enumerate(args):
                try:
                    args[j] = eval(arg) if isinstance(arg, str) else arg
                    exec(f"c{j} = args[j]")
                except:
                    pass
            n = max(round(n * gd), 1) if n > 1 else n
            idx = i+f if f == -1 else f
            print(f"[{i}]\t\t{idx}\t\t{module.__name__}\t\t{args}")
            module = module(n=n, c0=c0, c1=c1, c2=c2, c3=c3)
            print(module)

    def relevance(self, rel, alpha=None, beta=None, eps=None):
        pass