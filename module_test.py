import os
from KonanXAI.datasets import load_dataset
from KonanXAI.models import model_import
from KonanXAI.utils.data_convert import convert_tensor
from project.make_project import Project
# config_path = './project/example_gradcam/config_resnet_gradcam.yaml'
# config_path = './project/example_lime/config_resnet_lime.yaml'
config_path = './project/example_train/config_fgsm_resnet_train.yaml'
def config_test(config):
    project = Project(config)
    print(project.__dict__)

def dataLoader_test(config):
    conf = Project(config)
    dataset = load_dataset(conf.__dict__['framework'], data_path = conf.__dict__['data_path'],
                                    data_type = conf.__dict__['data_type'], resize = conf.__dict__['data_resize'], mode = conf.__dict__['project_type'])
    print(dataset)

def modelLoader_test(config):
    conf = Project(config)
    dataset = load_dataset(conf.__dict__['framework'], data_path = conf.__dict__['data_path'],
                                    data_type = conf.__dict__['data_type'], resize = conf.__dict__['data_resize'], mode = conf.__dict__['project_type'])
    model = model_import(conf.__dict__['framework'], conf.__dict__['source'], conf.__dict__['repo_or_dir'],
                                  conf.__dict__['model_name'], conf.__dict__['cache_or_local'], 
                                  conf.__dict__['weight_path'], conf.__dict__['cfg_path'], dataset.classes, conf.__dict__['model_algorithm'])
    print(model)
    
def run_test(config):
    project = Project(config)
    project.run()
    
# config_test(config_path)
# dataLoader_test(config_path)
# modelLoader_test(config_path)
run_test(config_path)