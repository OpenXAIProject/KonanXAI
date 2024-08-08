from ..config import Configuration
from ..utils import ModelType, PlatformType
import darknet
from .model import XAIModel
import torch
import torch.nn as nn
repository = {
    ModelType.ResNet50: 'pytorch/vision:v0.11.0',
    ModelType.VGG16: 'pytorch/vision:v0.11.0',
    ModelType.VGG19: 'pytorch/vision:v0.11.0',
    ModelType.EfficientNet_B0: 'pytorch/vision:v0.11.0.',
    #ModelType.Yolov4: '',
    ModelType.Yolov5s: 'ultralytics/yolov5',
}

# TODO 각 모델 폴더들은 플랫폼 별 실 모델 구현체로 사용하도록
# Pytorch, DTrain, Darknet 등...
def load_weights(weight_path:str, model_name:str, net):
    pt = torch.load(weight_path)
    state_dict = {}
    if 'yolo' in model_name:
        try:
            model = pt['model'].model.state_dict()
            for k, v in model.items():
                key = k if k.startswith('model.') else 'model.'+ k[:]
                state_dict[key] = v
            net.load_state_dict(state_dict)
        except:
            net = pt['ema'].float().fuse().eval()
    else:
        state_dict = pt
        net.load_state_dict(state_dict)
    return net

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
            net = load_weights(weight_path, model_name, net)
            # net.load_state_dict(state_dict)
        net.to(device)
        # full_backward를 위함 
        for name, module in net.named_modules():
            if isinstance(module, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
                module.inplace = False
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