from ..config import Configuration
from ..utils import ModelType, PlatformType
from ..lib.core import darknet
from .model import XAIModel
import torch
repository = {
    ModelType.ResNet50: 'pytorch/vision:v0.10.0',
    ModelType.VGG16: 'pytorch/vision:v0.10.0',
    ModelType.VGG19: 'pytorch/vision:v0.10.0',
    #ModelType.Yolov4: '',
    ModelType.Yolov5s: 'ultralytics/yolov5',
}

# TODO 각 모델 폴더들은 플랫폼 별 실 모델 구현체로 사용하도록
# Pytorch, DTrain, Darknet 등...
def load_weights(weight_path, model_name):
    pt = torch.load(weight_path)
    state_dict = {}
    if 'yolo' in model_name:
        pt = pt['model'].model.state_dict()
        for k, v in pt.items():
            key = k if k.startswith('model.') else 'model.'+ k[:]
            state_dict[key] = v
    else:
        state_dict = pt
    return state_dict

def load_model(config: Configuration, mtype: ModelType, platform: PlatformType,weight_path: str, **kwargs):
    # TODO 임시로 아래 내용 사용
    net = None
    if platform == PlatformType.Darknet:
        path = weight_path
        net = darknet.Network()
        net.load_model_custom(f"{path}.cfg", f"{path}.weights")
        model = XAIModel(config, mtype, platform, net)
    elif platform == PlatformType.Pytorch:
        # Torchhub
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if mtype not in repository:
            if 'repo' not in kwargs:
                raise Exception
            else:
                repo = kwargs['repo']
        else:
            repo = repository[mtype]
        source = 'github'
        if 'source' in kwargs:
            source = kwargs['source']
        model_name = mtype.name.lower()
        net = torch.hub.load(repo, model_name, source=source, **kwargs)
        if weight_path is not None:        
            state_dict = load_weights(weight_path, model_name)
            net.load_state_dict(state_dict)
        net.to(device)
        model = XAIModel(config, mtype, platform, net)
        for name, param in net.named_parameters():
            model.device = param.dtype
            break
    return model

def load_model_statedict(config: Configuration, mtype: ModelType, platform: PlatformType,weight_path: str, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mtype not in repository:
        if 'repo' not in kwargs:
            raise Exception
        else:
            repo = kwargs['repo']
    else:
        repo = repository[mtype]
    source = 'github'
    if 'source' in kwargs:
        source = kwargs['source']
    model_name = mtype.name.lower()  
    net = torch.hub.load(repo, model_name, source=source, **kwargs)
    state_dict = torch.load(weight_path)
    net = state_dict['ema']
    net.to(device)
    model = XAIModel(config, mtype, platform, net)
    for name, param in net.named_parameters():
        model.device = param.dtype
        break
    return model