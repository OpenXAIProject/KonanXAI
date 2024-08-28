import yaml
import os, sys
from KonanXAI.model_improvement.abn import ABN
from KonanXAI.model_improvement.trainer import Trainer
from KonanXAI.models.modifier.abn_resnet import make_attention_resnet50
from KonanXAI.model_improvement.domain_generalization import DomainGeneralization
from KonanXAI.models.modifier.abn_vgg import make_attention_vgg19
import torchvision.models as models
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import KonanXAI as XAI
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
class Configuration:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self._parser_config()        
        
    def _parser_config(self):
        self._public_parser()
        self._public_check_config()
        if self.project_type.lower() == 'explain':
            self._explain_parser()
            self._explain_algorithm_parser()
            self._explain_check_config()
        elif self.project_type.lower() == 'train':
            self._train_parser()
            self._train_check_config()
            
    def _public_parser(self):
        self.project_type = self.config['project']['project_type']
        self.save_path = self.config['project']['save_path']
        self.weight_path = self.config['project']['weight_path']
        self.cfg_path = self.config['project']['cfg_path']
        self.data_path = self.config['project']['data_path']
        self.data_resize = self.config['project']['data_resize']
        self.model_name = self.config['project']['model_name']
        self.framework = self.config['project']['framework']
        self.source = self.config['project']['source']
        self.repo_or_dir = self.config['project']['repo_or_dir']
        self.cache_or_local = self.config['project']['cache_or_local']
        self.data_type = self.config['project']['data_type']
        
    def _train_parser(self):
        self.epoch = self.config['project']['epoch']
        self.learning_rate = self.config['project']['learning_rate']
        self.batch_size = self.config['project']['batch_size']
        self.optimizer = self.config['project']['optimizer']
        self.loss_function = self.config['project']['loss_function']
        self.save_step = self.config['project']['save_step']
        self.improvement_algorithm = self.config['project']['improvement_algorithm']
        self.algorithm_name = self.improvement_algorithm['algorithm']
        self.transfer_weights = self.improvement_algorithm['transfer_weights']
        self.gpu_count = self.improvement_algorithm['gpu_count']
        
    def _explain_parser(self):
        self.explain_algorithm = self.config['project']['explain_algorithm']
        self.algorithm_name = list(self.explain_algorithm[0].keys())[0] 
        
    def _explain_algorithm_parser(self):
        cams = ['GradCAM','GradCAMpp','EigenCAM']
        lrps = ['LRP', 'LRPYolo']
        if self.algorithm_name.lower() in [cam.lower() for cam in cams]:
            self._gradcam_parser()
        
        elif self.algorithm_name.lower() in [lrp.lower() for lrp in lrps]:
            self._lrp_parser()
        
    def _gradcam_parser(self):
        self.config = {}
        self.config['target_layer'] = list(self.explain_algorithm[0].items())[0][1][0]['target_layer']
        self.config['algorithm'] = self.algorithm_name
    
    def _lrp_parser(self):
        self.config = {}
        self.config['algorithm'] = self.algorithm_name
        self.config['rule'] = rule = list(self.explain_algorithm[0].items())[0][1][0]['rule']
        self.config['yaml_path'] = self.cfg_path
        
    def _public_check_config(self):
        frameworks = ['torch', 'darknet']
        projects = ['train','explain']
        if self.project_type.lower() not in [project.lower() for project in projects]:
            msg = f"The type you entered is:'{self.project_type}' Supported types are: {projects}"
            raise Exception(msg)
        elif not isinstance(self.data_resize, (tuple,list)):
            raise Exception("Supported types are: 'tuple' or 'list'")
        elif self.framework.lower() not in [framework.lower() for framework in frameworks]:
            msg = f"The type you entered is:'{self.framework}' Supported types are: {frameworks}"
            raise Exception(msg)
        
    def _explain_check_config(self):
        attributions = ['GradCAM', 'GradCAMpp', 'EigenCAM', 'LRP', 'LRPYolo']
        if self.algorithm_name.lower() not in [attribution.lower() for attribution in attributions]:
            msg = f"The type you entered is:'{self.algorithm_name}' Supported types are: {attributions}"
            raise Exception(msg)
        
    def _train_check_config(self):
        improvement_algorithms = ['ABN', 'DomainGeneralization', 'Default']
        optimizers = ['Adam', 'SGD']
        loss_functions = ['CrossEntropyLoss', 'NLLLoss', 'MSELoss']
        if os.path.isdir(self.save_path) == False:
                os.makedirs(self.save_path) 
        # paser check                
        if self.algorithm_name.lower() not in [improvement_algorithm.lower() for improvement_algorithm in improvement_algorithms]:
            msg = f"The type you entered is:'{self.improvement_algorithm}' Supported types are: {improvement_algorithms}"
            raise Exception(msg)
        elif self.optimizer.lower() not in [optimizer.lower() for optimizer in optimizers]:
            msg = f"The type you entered is:'{self.optimizer}' Supported types are: {optimizers}"
            raise Exception(msg)
        elif self.loss_function.lower() not in [loss_function.lower() for loss_function in loss_functions]:
            msg = f"The type you entered is:'{self.loss_function}' Supported types are: {loss_functions}"
            raise Exception(msg)
        # optimizer
        if self.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam
        elif self.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD
        # loss function
        if self.loss_function.lower() == 'crossentropyloss':
            self.loss_function = nn.CrossEntropyLoss
        elif self.loss_function.lower() == 'nllloss':
            self.loss_function = F.nll_loss
        elif self.loss_function.lower() == 'mseloss':
            self.loss_function = nn.MSELoss
        # improvement algorithm
        
        if self.algorithm_name.lower() == 'abn':
            self.improvement_algorithm = ABN
            self.improvement_algorithm.name = 'abn'
            if self.model_name.lower().startswith("resnet"):
                self.make_model = make_attention_resnet50
            elif self.model_name.lower().startswith("vgg"):
                self.make_model = make_attention_vgg19
        elif self.algorithm_name.lower() == 'domaingeneralization':
            self.set_freq = self.improvement_algorithm['set_freq']
            self.target_layer = self.improvement_algorithm['target_layer']
            self.improvement_algorithm = DomainGeneralization
            self.improvement_algorithm.name = 'dg'            
            if self.model_name.lower().startswith("resnet"):
                self.make_model = models.resnet50
            elif self.model_name.lower().startswith("vgg"):
                self.make_model = models.vgg19
        elif self.algorithm_name.lower() == 'default':
            self.improvement_algorithm = Trainer
            self.improvement_algorithm.name = 'default'
            if self.model_name.lower().startswith("resnet"):
                self.make_model = models.resnet50
            elif self.model_name.lower().startswith("vgg"):
                self.make_model = models.vgg19
                
        if self.gpu_count >0:
            gpus = ""
            for i in range(self.gpu_count):
                gpus = gpus + f"{i},"
            gpus = gpus[:-1]
            self.gpu_count = gpus
        else:
            msg = f"The value you entered is:'{self.gpu_count}' The value must be greater than or equal to 1."
            raise Exception(msg)