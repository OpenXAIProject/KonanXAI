from ..config import Configuration
from ..utils import ModelType, PlatformType

class XAIModel:
    def __init__(self, config: Configuration, mtype: ModelType, platform: PlatformType, net: object):
        self.config = config
        self.mtype = mtype
        self.platform = platform
        self.net = net
        self.last_outputs = None

    def forward(self, data):
        out = None
        if self.platform == PlatformType.Darknet:
            out = self.net.forward_image(data)
        elif self.platform == PlatformType.DTrain:
            out = None
        elif self.platform == PlatformType.Pytorch:
            out = self.net(data)
        self.last_outputs = out
        
    # TODO - 임시 메서드 
    def backward(self):
        if self.platform == PlatformType.Darknet:
            self.net.backward()
        elif self.platform == PlatformType.DTrain:
            None
        elif self.platform == PlatformType.Pytorch:
            None

    # TODO - 여기서는 Backward 내용 준비
    def _set_delta(self):
        pass
        
    def get_nms_bbox(self):
        if self.platform == PlatformType.Darknet:
            pass
        elif self.platform == PlatformType.DTrain:
            pass
        elif self.platform == PlatformType.Pytorch:
            pass

