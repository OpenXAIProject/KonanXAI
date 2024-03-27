from .config import Configuration
from .utils import *
from .models import XAIModel, load_model

class XAI:
    def __init__(self):
        # Public variables
        self.config = None
        self.model: XAIModel = None
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


    #----------------------------------------------------
    # Algorithm Import

    #----------------------------------------------------
    # Explain
    def explain(self):
        pass