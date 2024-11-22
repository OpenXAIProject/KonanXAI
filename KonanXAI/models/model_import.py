from collections import OrderedDict
import torch
import torchvision
import torch.nn as nn
import urllib
from KonanXAI.models.modifier.abn_vgg import make_attention_vgg19
from KonanXAI.models.modifier.abn_resnet import make_attention_resnet50 
from collections import OrderedDict
import os, sys

# import 경로만 모아놔야 하는데..
from KonanXAI.models.hubconf import Dtrain, TorchGit, TorchLocal, Yolov5, Ultralytics, DarknetGit, DarknetLocal
from KonanXAI.models.modifier.dann_resnet import make_dann_resnet50
path = os.path.join(os.getcwd(), 'ultralytics')
sys.path.append(path)
from ultralytics.nn.modules.block import C2f
__all__ = ["model_import"]
#from KonanXAI._core import darknet

# torch/hub로 torchvision 모델 불러오려고 하니 hubconf.py 없어서 에러 생김
__version_of_torchvision__ = [
    'pytorch/vision:v0.11.0'
]

# torchvision.models에서 제공하는 모델 이름
__torchvision_models__ = [
    'efficientnet_b0',
    'resnet50'
]


__repository__ = [
    'ultralytics/yolov5',
    'ultralytics/ultralytics'
]

# 이건 각 source별로 모델 네이밍이 다를 거 같아 좀 생각해봐야..
__list_of_models__ = [
    'ResNet50',
    'VGG19',
    'EfficientNet-b0'
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_weight():
    pass

# def _parsing_git_url(git_url):

#     #return folder_tree
#     pass

def adjust_first_layer(model, pt):
    first_pt_key = list(pt.keys())[0]
    first_layer_key = list(model._modules)[0]
    batch, channel, height, width = pt[first_pt_key].shape
    in_channels = model._modules[first_layer_key].in_channels
    if in_channels == channel:
        return model
    else:
        in_channels = channel
        out_channels = model._modules[first_layer_key].out_channels
        kernel_size = model._modules[first_layer_key].kernel_size
        stride = model._modules[first_layer_key].stride
        padding = model._modules[first_layer_key].padding
        bias = model._modules[first_layer_key].bias
        model._modules[first_layer_key] = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = bias)
        return model
    
def adjust_last_layer(model, pt, num_classes):
    last_pt_key = list(pt.keys())[-2]
    last_layer_key = list(model._modules)[-1]
    features, chnnels = pt[last_pt_key].shape
    out_features = model._modules[last_layer_key].out_features
    if out_features == features:
        return model
    else:
        out_features = features
        out_channels = model._modules[_layer_key].out_channels
        kernel_size = model._modules[first_layer_key].kernel_size
        stride = model._modules[first_layer_key].stride
        padding = model._modules[first_layer_key].padding
        bias = model._modules[first_layer_key].bias
        model._modules[first_layer_key] = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = bias)
        return model




def torch_local_model_load(local_path, model_name, weight_path, num_classes, model_algorithm):
    local = TorchLocal(local_path, model_name, num_classes, model_algorithm)
    model = local._load(weight_path)
    #_get_file_tree(local_path)
    return model


# 그냥 git._load에서 download까지 한꺼번에 하게 할까 분리할까...?
def torch_git_repository_load(repo_or_dir, model_name, cache_or_local, weight_path):
    git = TorchGit(repo_or_dir, model_name)
    git._download_from_url(cache_or_local)
    model = git._load(weight_path)
    return model

# main인지 master인지 알아내야하는 코드가 필요하네?
# branch까지 기재하는걸로 할까?
def torch_model_load(
        source = None,
        repo_or_dir = None,
        model_name = None,
        cache_or_local = None,
        weight_path = None,
        num_classes = None,
        model_algorithm = None):
    
    # 다른 기능이 필요한게 있나?
    if source == 'torchvision':
        if weight_path == None:
            model = torch.hub.load(__version_of_torchvision__[0], model_name, num_classes = num_classes)
            model.model_name = model_name
            model.eval()
            return model
        else:
            pt = torch.load(weight_path)
            if model_algorithm == None:
                model = torch.hub.load(__version_of_torchvision__[0], model_name.lower(), num_classes = num_classes)
            
            elif model_algorithm.lower() in ["default", "domaingeneralization", 'fgsm']:
                model = torch.hub.load(__version_of_torchvision__[0], model_name.lower(), num_classes = num_classes)
            elif model_algorithm.lower() == 'abn':
                if "resnet" in model_name:
                    model = make_attention_resnet50(num_classes = num_classes)
                elif "vgg" in model_name:
                    model = make_attention_vgg19(num_classes= num_classes)
            
                    # model = make_dann_vgg19(num_classes= num_classes)
            if isinstance(pt, OrderedDict):
                state_dict = pt
            else:
                model_key = next(iter(model.state_dict()))
                state_dict = {}
                if 'module.' in model_key:
                    for k, v in pt['model_state_dict'].items():
                        key = 'module.'+ k 
                        state_dict[key] = v
                else:
                    for k, v in pt['model_state_dict'].items():
                        key = k[7:] if k.startswith('module.') else k
                        state_dict[key] = v
            
            adjust_first_layer(model, state_dict)

            model.load_state_dict(state_dict)
            model.model_name = model_name
            model.model_algorithm = model_algorithm
            model.output_size = num_classes
            model.eval()
            return model

    
    # elif source == 'torch/hub':
    #     model = torch.hub()

    # main branch인지 master branch인지 알아내야.. 
    # 일단 ultralytics/ultralytics 로 테스트 할 거니까 main으로 설정
    elif source == 'github':
        model = torch_git_repository_load(repo_or_dir, model_name, cache_or_local, weight_path)
        return model


    elif source == 'local':
        local_path = repo_or_dir
        model = torch_local_model_load(local_path, model_name, weight_path, num_classes, model_algorithm)
        return model
    
    elif source == 'ultralytics':
        from ultralytics import YOLO    
        model = YOLO(weight_path, task='detect').model
        for p in model.parameters():
            p.requires_grad_(True)
        model = model.to(device)
        model.float().fuse().eval()
        model.model_name = model_name
        model.model_algorithm = model_algorithm
        model.output_size = num_classes
        return model
    

def darknet_local_model_load(local_path, model_name, weight_path, cfg_path):
    local = DarknetLocal(local_path, model_name)
    model = local._load(weight_path, cfg_path)

    return model


# 그냥 git._load에서 download까지 한꺼번에 하게 할까 분리할까...?
def darknet_git_repository_load(repo_or_dir, model_name, cache_or_local, weight_path, cfg_path):
    model = DarknetGit(repo_or_dir, model_name)
    model._download_from_url(cache_or_local)
    model = model._load(weight_path, cfg_path)
    return model
    
    #return model   

def darknet_model_load(
        source = None,
        repo_or_dir = None,
        model_name = None, 
        cache_or_local = None, 
        weight_path = None,
        cfg_path = None):
    
    if source == 'github':
        model = darknet_git_repository_load(repo_or_dir, model_name, 
                                            cache_or_local, weight_path, cfg_path)

    elif source == 'local':
        local_path = repo_or_dir
        model = darknet_local_model_load(local_path, model_name, 
                                         weight_path, cfg_path)
        
    return model
        


def dtrain_model_load(model_name, weight_path):
    model_object = Dtrain(model_name, weight_path)
    model = model_object._load()

    return model


def model_import(
        framework = 'torch',
        source = 'github',
        repo_or_dir = None,
        model_name = None,
        cache_or_local = None,
        weight_path = None,
        cfg_path = None,
        num_classes = None,
        model_algorithm = None):
    '''
    - posibble framework: 'torch', 'darknet', 'dtrain'
    - source: 'torchvision', 'torch/hub', 'github', 'local'
    - repo_or_dir: 'repository_owner/repository_name' or 'user model path'
    - model_name: ...
    - cache_or_local: 'cache'에 저장 혹은 'local_path'에 저장
    - weight_path: ...
    - cfg_path: 'local cfg file path'

    - example for torchvision models load
        framework = 'torch'
        source = 'torchvision'
        repo_or_dir = None
        model_name = 'ResNet50'
        weight_path = 'user weight path' 

    - example for torch.hub models load
    - example for github repository load
    '''

    if framework == 'torch':
        model = torch_model_load(source = source, 
                                  repo_or_dir = repo_or_dir, 
                                  model_name = model_name, cache_or_local = cache_or_local, 
                                  weight_path = weight_path, num_classes= num_classes, model_algorithm = model_algorithm)
        for name, module in model.named_modules():
            if isinstance(module, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
                module.inplace = False  
        model.to(device)  
    elif framework == 'darknet':
        model = darknet_model_load(source = source, 
                                    repo_or_dir = repo_or_dir,
                                    model_name = model_name, cache_or_local = cache_or_local,
                                    weight_path = weight_path,
                                    cfg_path = cfg_path)
    elif framework == 'dtrain':
        model = dtrain_model_load(model_name = model_name, weight_path = weight_path)

    return model