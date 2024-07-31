import torch
import torchvision

import urllib
import git

import os

# import 경로만 모아놔야 하는데..
from KonanXAI.models.hubconf import Git, Yolov5, Ultralytics

import KonanXAI._core


# torch/hub로 torchvision 모델 불러오려고 하니 hubconf.py 없어서 에러 생김
__version_of_torchvision__ = [
    'pytorch/vision:v0.10.0'
]

__torchvision_models__ = [
    'efficientnet_b0'
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

def load_weight():
    pass

# def _parsing_git_url(git_url):

#     #return folder_tree
#     pass




def _local_model_load(local_path, model_name):
    model = Ultralytics(local_path, model_name)
    model = model._load()
    #_get_file_tree(local_path)
    return model


# 그냥 git._load에서 download까지 한꺼번에 하게 할까 분리할까...?
def _git_repository_load(repo_or_dir, model_name, cache_or_local):
    git = Git(repo_or_dir, model_name)
    git._download_from_url(cache_or_local)
    model = git._load()
    
    return model



# main인지 master인지 알아내야하는 코드가 필요하네?
# branch까지 기재하는걸로 할까?
def _torch_model_load(
        source = None,
        repo_or_dir = None,
        model_name = None,
        cache_or_local = None,
        weight_path = None):
    
    
    # 다른 기능이 필요한게 있나?
    if source == 'torchvision':
        model = torchvision.models.get_model(model_name)
        return model
    
    # elif source == 'torch/hub':
    #     model = torch.hub()

    # main branch인지 master branch인지 알아내야.. 
    # 일단 ultralytics/ultralytics 로 테스트 할 거니까 main으로 설정
    elif source == 'github':
        model = _git_repository_load(repo_or_dir, model_name, cache_or_local)
        return model


    elif source == 'local':
        local_path = repo_or_dir
        model_name = model_name
        model = _local_model_load(local_path, model_name)
        return model


    

def _darknet_model_load():
    pass

def _dtrain_model_load():
    pass




def model_import(
        framework = 'torch',
        source = 'github',
        repo_or_dir = None,
        model_name = None,
        cache_or_local = None,
        weight_path = None):
    '''
    - posibble framework: 'torch', 'darknet', 'dtrain'
    - source: 'torchvision', 'torch/hub', 'github', 'local'
    - repo_or_dir: 'repository_owner/repository_name' or 'user model path'
    - model_name: ...
    - cache_or_local: 'cache'에 저장 혹은 'local_path'에 저장
    - weight_path: ...
    
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
        model = _torch_model_load(source = source, 
                                  repo_or_dir = repo_or_dir, 
                                  model_name = model_name, cache_or_local = cache_or_local, 
                                  weight_path = weight_path)




    return model