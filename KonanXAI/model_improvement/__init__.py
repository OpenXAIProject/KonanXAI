from KonanXAI.models.modifier.abn_resnet import make_attention_resnet50
from KonanXAI.models.modifier.abn_vgg import make_attention_vgg19
from KonanXAI.models.modifier.dann_resnet import make_dann_resnet50
from .abn import *
from .dann_grad import *
from . dann import *
from .domain_generalization import *
from .fgsm import *
from .trainer import *
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
# out = model, optimizer, criterion, trainer
def load_train(config):    
    value, hyperparm = config_parser(config)
    model, trainer_type = load_algorithm(value)
    optimize = load_optimizer(value)
    criterion = load_loss_function(value)
    gpu_count = set_gpu(value)
    return model, optimize, criterion, trainer_type, gpu_count, hyperparm

def config_parser(config):
    config_item = {}
    hyperparm = {}
    lower_parser = ['optimizer', 'loss_function', 'algorithm']
    hyperparam_parser = ['lr', 'batch', 'epoch']
    for key, value in config.items():
        if isinstance(value, dict):
            for key_, value_ in config[key].items():
                if key_ in lower_parser:
                    config_item[key_] = value_.lower()
                else:
                    config_item[key_] = value_
        if key in lower_parser:
            config_item[key] = value.lower()
        else:
            if key in hyperparam_parser:
                hyperparm[key] = value
            config_item[key] = value
    return config_item, hyperparm
            
def load_algorithm(value):
    improvement_algorithms = ['ABN', 'DomainGeneralization', 'DANN', 'DANN_GRAD', 'Default','FGSM']
    if value['algorithm'] not in [improvement_algorithm.lower() for improvement_algorithm in improvement_algorithms]:
        msg = f"The type you entered is:'{value['algorithm']}' Supported types are: {improvement_algorithms}"
        raise Exception(msg)
    if value['algorithm'] == 'abn':
        model_algorithm = value['algorithm']
        improvement_algorithm = ABN
        model =_make_abn_model(value['model_name'])
    elif value['algorithm'] == 'domaingeneralization':
        model_algorithm =value['algorithm']
        improvement_algorithm = DomainGeneralization
        improvement_algorithm.frequency = value['set_freq']
        improvement_algorithm.target_layer = value['target_layer']
        model =_make_model(value['model_name'])
    elif value['algorithm'] == 'default':
        model_algorithm = value['algorithm']
        improvement_algorithm = Trainer
        model =_make_model(value['model_name'])
    elif value['algorithm'] == 'fgsm':
        model_algorithm = value['algorithm']
        improvement_algorithm = FGSM
        improvement_algorithm.epsilon = value['epsilon']
        improvement_algorithm.alpha = value['alpha']
        model =_make_model(value['model_name'])
    elif value['algorithm'] == 'dann':
        model_algorithm = value['algorithm']
        improvement_algorithm = DANN
        model = _make_dann_model(value['model_name'])
    elif value['algorithm'] == 'dann_grad':
        model_algorithm = value['algorithm']
        improvement_algorithm = DANN_GRAD
        improvement_algorithm.target_layer = value['target_layer']
        model = _make_dann_model(value['model_name'])
    improvement_algorithm.model_algorithm = model_algorithm
    improvement_algorithm.transfer_weights = value['transfer_weights']
    return model, improvement_algorithm

def load_optimizer(value):
    optimizers = ['Adam', 'SGD']
    if value['optimizer'] not in [optimizer.lower() for optimizer in optimizers]:
        msg = f"The type you entered is:'{value['optimizer']}' Supported types are: {optimizers}"
        raise Exception(msg)
    if value['optimizer'] == 'adam':
        return optim.Adam
    elif value['optimizer'] == 'sgd':
        return optim.SGD

def load_loss_function(value):
    loss_functions = ['CrossEntropyLoss', 'NLLLoss', 'MSELoss']
    if value['loss_function'] not in [loss_function.lower() for loss_function in loss_functions]:
        msg = f"The type you entered is:'{value['loss_function']}' Supported types are: {loss_functions}"
        raise Exception(msg)
    if value['loss_function'] == 'crossentropyloss':
        return nn.CrossEntropyLoss
    elif value['loss_function'] == 'nllloss':
        return F.nll_loss
    elif value['loss_function'] == 'mseloss':
        return nn.MSELoss

def set_gpu(value):
    if value['gpu_count'] >0:
        gpus = []
        for i in range(value['gpu_count']):
            gpus.append(i)
        return gpus
    else:
        msg = f"The value you entered is:'{value['gpu_count']}' The value must be greater than or equal to 1."
        raise Exception(msg)               
    
def _make_model(model_name):
    if model_name.startswith("resnet"):
        return models.resnet50
    elif model_name.startswith("vgg"):
        return models.vgg19
        
def _make_abn_model(model_name):
    if model_name.startswith("resnet"):
        return make_attention_resnet50
    elif model_name.startswith("vgg"):
        return make_attention_vgg19
        
def _make_dann_model(model_name):
    if model_name.startswith("resnet"):
        return make_dann_resnet50
    elif model_name.startswith("vgg"):
        raise Exception("Not Supported")