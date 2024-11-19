from typing import Tuple, Union, Callable, List, TypeVar, Sequence,Literal
from torch import Tensor
from torch.nn.modules import Module
from abc import ABC
import copy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from KonanXAI.utils.heatmap import linear_transform
TensorOrTupleOfTensors = Union[Tensor, Tuple[Tensor]]
ForwardArgumentExtractor = Callable[[TensorOrTupleOfTensors], TensorOrTupleOfTensors]
TargetLayer = Union[str, Module]
TargetLayerOrListOfTargetLayers = Union[TargetLayer, List[TargetLayer]]
__all__= ["format_into_tuple", "format_into_tuple_all", "ZeroBaselineFunction", "postprocessed_ig", "postprocessed_lime", "postprocessed_guided", "heatmap_postprocessing"]
T = TypeVar('T')
def format_into_tuple(obj: T) -> Tuple[T]:
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return tuple(obj)
    elif isinstance(obj, type(None)):
        return ()
    return (obj,)

def format_into_tuple_all(**kwargs):
    return {k: format_into_tuple(v) for k, v in kwargs.items()}

class UtilFunction(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def copy(self):
        return copy.copy(self)

    def set_kwargs(self, **kwargs):
        clone = self.copy()
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(clone, k, v)
        return clone

    def __call__(self, inputs: torch.Tensor):
        return NotImplementedError

    def get_tunables(self):
        return {}
class BaselineFunction(UtilFunction):
    """
    A base class for baseline functions used in attribution methods. Baseline functions are 
    used to define a reference or baseline value against which attributions are compared.
    This is typically used to understand the effect of different inputs on the model's predictions.

    Notes:
        - `BaselineFunction` is intended to be subclassed. Concrete baseline functions should 
          inherit from this class and implement the actual baseline logic.
        - Subclasses can override the `__init__` method to accept additional parameters required 
          for their specific baseline operations.
    """
    
    def __init__(self, *args, **kwargs):
        pass
class ZeroBaselineFunction(BaselineFunction):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs: torch.Tensor):
        return torch.zeros_like(inputs)
    
def postprocessed_ig(attr, dim):
    # positive = np.clip(attr, 0, 1)
    # gray_ig = np.average(positive, axis=2)
    # linear_attr = linear_transform(gray_ig, 99, 0, 0.0, plot_distribution=False)
    # linear_attr = np.expand_dims(linear_attr, 2) * [0, 255, 0]
    # ig_image = np.array(linear_attr, dtype=np.uint8)
    # res = torch.tensor(ig_image)
    res = torch.tensor(attr)
    # plt.imshow(res.clamp(min=0).sum(dim).unsqueeze(0).squeeze(0))
    poold = res.abs().max(dim)[0].unsqueeze(0)
    # poold = res.abs().max(dim)[0].unsqueeze(0) 
    return poold

def postprocessed_lime(attr,dim):
    res = torch.tensor(attr)
    poold = res.abs().max(dim)[0].unsqueeze(0)
    return poold

def postprocessed_guided(attr, dim, img_size):
    heatmap, guided = attr
    heatmap = F.interpolate(heatmap, img_size, mode='bilinear').detach().cpu()
    heatmap_mask = np.transpose(heatmap.squeeze(0).cpu().numpy(),(1,2,0))
    res = heatmap_mask*guided
    res = torch.tensor(np.transpose(res,(2,0,1))).unsqueeze(0)
    poold = res.pow(2).sum(dim).sqrt()
    # poold = res.clamp(min=0).sum(dim)
    # norm = normalize_heatmap(poold)
    return poold

def deprocess_images(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return img

def heatmap_postprocessing(algorithm_name, img_size, heatmap):
    if isinstance(heatmap, (tuple, list)):
        if "guided" in algorithm_name:
            heatmap = postprocessed_guided((heatmap[0][0],heatmap[1][0][0]),1, img_size)
        else:
            heatmap = heatmap[0]
    if algorithm_name == 'ig':
        heatmap = postprocessed_ig(heatmap,2)
    elif len(heatmap.shape) == 4:
        heatmap = F.interpolate(heatmap, img_size, mode='bilinear').detach().cpu().squeeze(0)
    elif len(heatmap.shape) == 2 and isinstance(heatmap,np.ndarray):
        heatmap = torch.tensor(heatmap).unsqueeze(0)
    elif algorithm_name == 'lime':
        heatmap = postprocessed_lime(heatmap, 2)
    return heatmap