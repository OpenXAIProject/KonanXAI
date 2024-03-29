from .config import Configuration
from .utils import *
from .models import XAIModel, load_model
from .datasets import *
from .lib import kernel
from .lib.algorithm import *

class XAI:
    def __init__(self):
        # Public variables
        self.config = None
        self.model: XAIModel = None
        self.dataset = None
        self.algorithm = None

    #----------------------------------------------------
    # Model Import 
    def load_model_support(self, mtype: ModelType, platform: PlatformType, pretrained=False):
        config = Configuration()
        self.model = load_model(config, mtype, platform)

    def load_model_custom(self, ):
        pass
    
    #----------------------------------------------------
    # Dataset Import
    def load_dataset_support(self, dtype: DatasetType, maxlen=-1, shuffle=False):
        # Test Code
        self.dataset = Custom()
    
    def load_dataset_custom(self, image_path, image_ext, label_path=None, label_ext=None, shuffle=False):
        pass

    #----------------------------------------------------
    # Algorithm Import
    def set_explain_mode(self, lists: list[ExplainType]):
        self.algorithm = lists

    #----------------------------------------------------
    # Explain
    def explain(self) -> kernel.ExplainData:
        return kernel.request_algorithm(self)