from .api import *
from .dict import Dict
from .yolo import *

class Layer:
    def __init__(self, layer_dict: Dict, net_w: int, net_h: int):
        self.idx = layer_dict['idx']
        self.type = layer_dict['type']
        self.inputs = layer_dict['inputs']
        self.w = layer_dict['w']
        self.h = layer_dict['h']
        self.c = layer_dict['c']
        self.n = layer_dict['n']
        self.outputs = layer_dict['outputs']
        self.out_w = layer_dict['out_w']
        self.out_h = layer_dict['out_h']
        self.out_c = layer_dict['out_c']
        self.output = layer_dict['output']
        self.delta = layer_dict['delta']
        self.biases = layer_dict['biases']
        self.new_coords = layer_dict['new_coords']
        self.classes = layer_dict['classes']
        self.mask = layer_dict['mask']
        self.net_w = net_w
        self.net_h = net_h
        
    def _get_ptr(self, ptr, n) -> list:
        result = []
        for i in range(n):
            result.append(ptr[i])
        return result

    def get_output(self) -> list:
        return self._get_ptr(self.output, self.outputs)
        
    def get_delta(self) -> list:
        return self._get_ptr(self.delta, self.outputs)
    
    def set_delta(self, data: list):
        assert len(data) == self.outputs, "The length of delta is different from the length of data"
        for i in range(self.outputs):
            self.delta[i] = data[i]
            # self.delta[i].contents

    def get_bboxes(self, batch=0, threshold=0.5) -> list[BBox]:
        assert self.type == LAYER_TYPE.YOLO, "This layer is not 'YOLO' layer"
        bboxes = []
        out = self.get_output()
        for i in range(self.out_w * self.out_h):
            px = i % self.out_w
            py = i // self.out_w
            for n in range(self.n):
                location = n * self.out_w * self.out_h + i
                obj_idx = entry_index(self, batch, location, 4)
                objectness = out[obj_idx]
                if objectness > threshold:
                    entry = entry_index(self, batch, location, 0)
                    bbox = get_yolo_box(self, self.mask[n], entry, px, py, self.net_w, self.net_h, threshold)
                    bboxes.append(bbox)
        return bboxes
    
    def get_class_probs(self, out: list, entry: int, batch: int=0) -> list[float]:
        assert self.type == LAYER_TYPE.YOLO, "This layer is not 'YOLO' layer"
        probs = []
        stride = self.out_w * self.out_h
        for i in range(self.classes):
            probs.append(out[entry + (i + 5) * stride])
        return probs

