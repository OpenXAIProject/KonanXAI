from .api import *
from .image import Image
from .dict import Dict
from .layer import Layer
class Network:
    def __init__(self):
        pass
    
    # Load Darknet Model Custom
    def load_model_custom(self, cfg_path: str, weights_path: str, clear: int=0, batch: int=0):
        # if batch == 0:
        #     cfg_path = cfg_path.encode("ascii")
        #     weights_path = weights_path.encode("ascii")
            # self.model_ptr = load_net(cfg_path, weights_path, clear)
        # else:
        cfg_path = cfg_path.encode("ascii")
        weights_path = weights_path.encode("ascii")
        self.model_ptr = load_net_custom(cfg_path, weights_path, clear, batch)
        self._get_network_info()
        self._get_network_layers()
            
    def __del__(self):
        self.free()

    def free(self):
        if hasattr(self, "model_ptr"):
            free_network_ptr(self.model_ptr)
            del self.model_ptr

    def _get_network_info(self):
        dict_ptr = get_network_info(self.model_ptr)
        info = Dict(dict_ptr)
        self.n = info['n']
        self.w = info['w']
        self.h = info['h']
        self.c = info['c']
        del dict_ptr

    def _get_network_layers(self):
        self.layers: list[Layer] = []
        dict_ptr = get_network_layers(self.model_ptr)
        layers = Dict(dict_ptr)
        for _, value in layers.items():
            self.layers.append(Layer(value, self.w, self.h))
        del dict_ptr

    def forward_image(self, image: Image):
        network_forward_image(self.model_ptr, image.data())
    
    def backward(self):
        network_backward(self.model_ptr)

    def zero_grad(self):
        network_zero_delta(self.model_ptr)
    
    def predict_using_gradient_hook(self, image: Image) -> Dict:
        dict_ptr = network_predict_using_gradient_hook(self.model_ptr, image.data())
        return Dict(dict_ptr)
    
    def get_network_boxes(self, image_width, image_height, thresh=.5, hier_thresh=.5):
        pnum = ct.pointer(ct.c_int(0))
        detection = get_network_boxes(self.model_ptr, image_width, image_height, thresh, hier_thresh, None, 0, pnum, 0)
        return detection, pnum[0]