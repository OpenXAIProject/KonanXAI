from typing import Tuple, Union, Callable, List, TypeVar, Sequence,Literal
from torch import Tensor
from torch.nn.modules import Module
from abc import ABC
import copy
import torch
TensorOrTupleOfTensors = Union[Tensor, Tuple[Tensor]]
ForwardArgumentExtractor = Callable[[TensorOrTupleOfTensors], TensorOrTupleOfTensors]
TargetLayer = Union[str, Module]
TargetLayerOrListOfTargetLayers = Union[TargetLayer, List[TargetLayer]]

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