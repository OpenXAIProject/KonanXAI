import torch
import torchvision

import urllib
import git

import KonanXAI._core

#__all__   # model들 리스트 쓰기                       

def _save_in_cache():
    pass

def _save_from_cache():  # torchvision or torch/hub 모델 저장할 때
    pass

def _git_clone(git_url = None, local_path_to_load = None):
    git.Git(local_path_to_load).clone(git_url)
    


def _torch_model_load(
        source = None,
        repo_or_dir = None,
        model_name = None,
        local_path_to_load = None,
        weight_path = None):
    
    
    
    # if source == 'torchvision':
    #     model = torchvision.models('') #.. model 이름을 속성으로 불러와야되네..
    #     return model
    
    # elif source == 'torch/hub':
    #     model = torch.hub()

    # 일단 cache로 저장하고 필요에 따라 로컬도 저장하게 해야할텐데 귀찮네
    if source == 'github':
        git_url = 'github.com/' + repo_or_dir
        _git_clone(git_url = git_url, local_path_to_load = local_path_to_load)


    

def _darknet_model_load():
    pass

def _dtrain_model_load():
    pass




def model_import(
        framework = 'torch',
        source = 'github',
        repo_or_dir = None,
        model_name = None,
        local_path_to_load = None,
        weight_path = None):
    '''
    - posibble framework: 'torch', 'darknet', 'dtrain'
    - source: 'torchvision', 'torch/hub', 'github', 'local'
    - repo_or_dir: 'repository_owner/repository_name' or 'user model path'
    - model_name: ...
    - local_path_to_load: path which users want to save the model
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
                                  model_name = model_name, local_path_to_load = local_path_to_load, 
                                  weight_path = weight_path)




    return model