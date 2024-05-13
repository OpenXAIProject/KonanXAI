from ..config import Configuration
from ..utils import ModelType, PlatformType
from ..lib.core import darknet
from .model import XAIModel

repository = {
    ModelType.ResNet50: 'pytorch/vision:v0.10.0',
    ModelType.VGG16: 'pytorch/vision:v0.10.0',
    ModelType.VGG19: 'pytorch/vision:v0.10.0',
    #ModelType.Yolov4: '',
    ModelType.Yolov5s: 'ultralytics/yolov5',
}

# TODO 각 모델 폴더들은 플랫폼 별 실 모델 구현체로 사용하도록
# Pytorch, DTrain, Darknet 등...

def load_model(config: Configuration, mtype: ModelType, platform: PlatformType, **kwargs):
    # TODO 임시로 아래 내용 사용
    net = None
    if platform == PlatformType.Darknet:
        # base = "./ckpt/darknet/yolov4tiny/"
        # name = "1175"
        base = "./ckpt/darknet/yolov4_5class/"
        name = "1093"
        path = base + name
        net = darknet.Network()
        net.load_model_custom(f"{path}.cfg", f"{path}.weights")
    elif platform == PlatformType.Pytorch:
        import torch
        # Torchhub
        if mtype not in repository:
            if 'repo' not in kwargs:
                raise Exception
            else:
                repo = kwargs['repo']
        else:
            repo = repository[mtype]
        
        model_name = mtype.name.lower()
        net = torch.hub.load(repo, model_name)
    model = XAIModel(config, mtype, platform, net)
    return model