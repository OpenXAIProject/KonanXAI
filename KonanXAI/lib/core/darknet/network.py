from .api import *
from .image import Image
from .dict import Dict

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
            
    def __del__(self):
        self.free()

    def free(self):
        if hasattr(self, "model_ptr"):
            free_network_ptr(self.model_ptr)
            del self.model_ptr

    def forward(self, x):
        pass
    
    def backward(self, delta):
        pass
    
    def predict_using_gradient_hook(self, image: Image) -> Dict:
        dict_ptr = network_predict_using_gradient_hook(self.model_ptr, image.data())
        return Dict(dict_ptr)
    
    def get_network_boxes(self, image_width, image_height, thresh=.5, hier_thresh=.5):
        pnum = ct.pointer(ct.c_int(0))
        detection = get_network_boxes(self.model_ptr, image_width, image_height, thresh, hier_thresh, None, 0, pnum, 0)
        return detection, pnum[0]