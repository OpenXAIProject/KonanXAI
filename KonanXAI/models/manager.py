from ..config import Configuration
from ..utils import ModelType, PlatformType
from ..lib.core import darknet
from .model import XAIModel

# TODO 각 모델 폴더들은 플랫폼 별 실 모델 구현체로 사용하도록
# Pytorch, DTrain, Darknet 등...

def load_model(config: Configuration, mtype: ModelType, platform: PlatformType):
    # TODO 임시로 아래 내용 사용
    net = None
    if platform == PlatformType.Darknet:
        base = "./ckpt/darknet/yolov4tiny/"
        name = "1175"
        path = base + name
        net = darknet.Network()
        net.load_model_custom(f"{path}.cfg", f"{path}.weights")
        
    model = XAIModel(config, mtype, platform, net)
    return model
