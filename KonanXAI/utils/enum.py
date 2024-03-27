import enum

class ModelType(enum.Enum):
    Custom          = enum.auto()
    VGG16           = enum.auto()
    VGG19           = enum.auto()
    ResNet50        = enum.auto()
    EfficientNetB0  = enum.auto()
    Yolov4          = enum.auto()
    Yolov4Tiny      = enum.auto()
    Yolov5s         = enum.auto()
    
class PlatformType(enum.Enum):
    Pytorch         = enum.auto()
    Darknet         = enum.auto()
    DTrain          = enum.auto()
    
class DatasetType(enum.Enum):
    COCO            = enum.auto()
    
class ExplainType(enum.Enum):
    GradCAM         = enum.auto()
    GradCAMpp       = enum.auto()
    LRP             = enum.auto()
    IG              = enum.auto()

class LRPRule(enum.Enum):
    Epsilon         = enum.auto()
    AlphaBeta       = enum.auto()
    