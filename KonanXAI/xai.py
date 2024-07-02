from .config import Configuration
from .utils import *
from .models import XAIModel, load_model, load_model_statedict
from .datasets import *
from .lib import kernel
from .lib.attribution import *

class XAI:
    def __init__(self):
        # Public variables
        self.config = None
        self.model: XAIModel = None
        self.dataset = None
        self.algorithm = None
        self.mtype = None

    #----------------------------------------------------
    # Model Import 
    def load_model_support(self, mtype: ModelType, platform: PlatformType,weight_path: str=None, pretrained=False, **kwargs):
        config = Configuration()
        self.model = load_model(config, mtype, platform, weight_path, pretrained = pretrained, **kwargs)

    def load_model_custom(self, mtype: ModelType, platform: PlatformType,weight_path: str=None, pretrained=False, **kwargs):
        #TODO: 여기에는 커스텀 모델 및 다양한 모델구조를 불러올 수 있도록 구현
        config = Configuration()
        self.model = load_model_statedict(config, mtype, platform, weight_path, pretrained = pretrained, **kwargs)
    
    #----------------------------------------------------
    # Dataset Import
    def load_dataset_support(self, dtype: DatasetType, maxlen=-1,path:str=None,fit_size: tuple=None):
        # Test Code
        self.dataset = globals().get(dtype.name)(path)# 다시 생각하기, 조건문으로 할지..
        self.dataset.fit = fit_size
        
    def load_dataset_custom(self, image_path, image_ext, label_path=None, label_ext=None, shuffle=False):
        pass

    #----------------------------------------------------
    # Algorithm Import
    def set_explain_mode(self, lists: list[ExplainType]):
        self.algorithm = lists

    #----------------------------------------------------
    # Explain
    def explain(self, target_layer = None) -> kernel.ExplainData:
        self.model.target_layer = target_layer
        return kernel.request_algorithm(self)