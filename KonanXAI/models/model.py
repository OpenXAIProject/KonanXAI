from ..config import Configuration
from ..utils import ModelType, PlatformType

class XAIModel:
    def __init__(self, config: Configuration, mtype: ModelType, platform: PlatformType, net: object):
        self.config = config
        self.mtype = mtype
        self.platform = platform
        self.net = net