from typing import Tuple, Union, Callable, List, TypeVar, Sequence,Literal
from torch import Tensor
from torch.nn.modules import Module

TensorOrTupleOfTensors = Union[Tensor, Tuple[Tensor]]
ForwardArgumentExtractor = Callable[[TensorOrTupleOfTensors], TensorOrTupleOfTensors]
TargetLayer = Union[str, Module]
TargetLayerOrListOfTargetLayers = Union[TargetLayer, List[TargetLayer]]
ExplanationType = Literal["attribution"]
T = TypeVar('T')
def format_into_tuple(obj: T) -> Tuple[T]:
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return tuple(obj)
    elif isinstance(obj, type(None)):
        return ()
    return (obj,)

def format_into_tuple_all(**kwargs):
    return {k: format_into_tuple(v) for k, v in kwargs.items()}
