import yaml
import os, sys
from KonanXAI.attribution.integrated_gradient import IG
from KonanXAI.attribution.kernel_shap import KernelShap
from KonanXAI.attribution.layer_wise_propagation.lrp import LRP
from KonanXAI.attribution.layer_wise_propagation.lrp_yolo import LRPYolo
from KonanXAI.attribution import GradCAM, GradCAMpp, EigenCAM, GuidedGradCAM
from KonanXAI.attribution import Gradient, GradientxInput, SmoothGrad, DeepLIFT
from KonanXAI.attribution.lime_image import LimeImage
from KonanXAI.evaluation.pixel_flipping import AbPC
from KonanXAI.evaluation.sensitivity import Sensitivity
from KonanXAI.model_improvement.dann import DANN
from KonanXAI.model_improvement.dann_grad import DANN_GRAD
from KonanXAI.model_improvement.fgsm import FGSM
from KonanXAI.model_improvement.abn import ABN
from KonanXAI.model_improvement.trainer import Trainer
from KonanXAI.models.modifier.abn_resnet import make_attention_resnet50
from KonanXAI.model_improvement.domain_generalization import DomainGeneralization
from KonanXAI.models.modifier.abn_vgg import make_attention_vgg19
import torchvision.models as models

from KonanXAI.models.modifier.dann_resnet import make_dann_resnet50
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
            self._explain_check_config()
        elif self.project_type.lower() == 'train':
            self._train_parser()
            self._train_check_config()
        elif self.project_type == 'evaluation':
            self._eval_parser()
            self._explain_check_config()
            self._evaluation_check_config()
            
    def _public_parser(self):
        self.project_type = self.config['head']['project_type']
        self.save_path = self.config['head']['save_path']
        self.weight_path = self.config['head']['weight_path']
        self.cfg_path = self.config['head']['cfg_path']
        self.data_path = self.config['head']['data_path']
        self.data_resize = self.config['head']['data_resize']
        self.model_name = self.config['head']['model_name']
        self.framework = self.config['head']['framework']
        self.source = self.config['head']['source']
        self.repo_or_dir = self.config['head']['repo_or_dir']
        self.cache_or_local = self.config['head']['cache_or_local']
        self.data_type = self.config['head']['data_type']
        
    def _train_parser(self):
        lower_parser = ['optimizer', 'loss_function', 'algorithm']
        for key, value in self.config['train'].items():
            if isinstance(value, dict):
                for key_, value_ in self.config['train'][key].items():
                    if key_ in lower_parser:
                        key_ = 'algorithm_name' if key_ in 'algorithm' else key_ 
                        setattr(self, key_, value_.lower())
                    else:
                        setattr(self, key_, value_)
            if key in lower_parser:
                setattr(self, key, value.lower())
            else:
                setattr(self, key, value)
        
        # self.epoch = self.config['train']['epoch']
        # self.learning_rate = self.config['train']['learning_rate']
        # self.batch_size = self.config['train']['batch_size']
        # self.optimizer = self.config['train']['optimizer'].lower()
        # self.loss_function = self.config['train']['loss_function'].lower()
        # self.save_step = self.config['train']['save_step']
        # self.improvement_algorithm = self.config['train']['improvement_algorithm']
        # self.algorithm_name = self.improvement_algorithm['algorithm'].lower()
        # self.transfer_weights = self.improvement_algorithm['transfer_weights']
        # self.gpu_count = self.improvement_algorithm['gpu_count']
        
    def _explain_parser(self):
        self.explains = self.config['explain']
        self.model_algorithm = self.explains['model_algorithm'].lower()
        self.algorithm_name = self.explains['algorithm'].lower()
        self.config = {}
        lower_parser = ['rule','algorithm']
        if 'lrp' in self.algorithm_name:
            self.config['yaml_path'] = self.cfg_path
        for key, value in self.explains.items():
            if key in lower_parser:
                self.config[key] = value.lower()
            else:
                self.config[key] = value
                
    def _eval_parser(self):
        self.explains = self.config['evaluation']
        self.model_algorithm = self.explains['model_algorithm'].lower()
        self.algorithm_name = self.explains['algorithm'].lower()
        self.config = {}
        lower_parser = ['rule','algorithm','metric']
        for key, value in self.explains.items():
            if key in lower_parser:
                self.config[key] = value.lower()
            else:
                self.config[key] = value
    # def _explain_algorithm_parser(self):
    #     self.config = {}
    #     lower_parser = ['rule','algorithm']
    #     for key, value in self.explains.items():
    #         if key in lower_parser:
    #             self.config[key] = value.lower()
    #         else:
    #             self.config[key] = value
    #     if self.algorithm_name in [cam.lower() for cam in cams]:
    #         self._gradcam_parser()
        
    #     elif self.algorithm_name in [lrp.lower() for lrp in lrps]:
    #         self._lrp_parser()
            
    #     elif self.algorithm_name == "ig":
    #         self._ig_parser()
    #     elif self.algorithm_name == "lime":
    #         self._lime_parser()
    #     elif self.algorithm_name == "kernelshap":
    #         self._kernelshap_parser()    
            
    # def _gradcam_parser(self):
    #     self.config['target_layer'] = self.explains['target_layer']
    #     self.config['algorithm'] = self.algorithm_name
    
    # def _lrp_parser(self):
    #     self.config['algorithm'] = self.algorithm_name
    #     self.config['rule'] = self.explains['rule'].lower()
    #     if self.config['rule'] == "alphabeta":
    #         self.config['alpha'] = self.explains['alpha']
    #     self.config['yaml_path'] = self.cfg_path
    
    # def _ig_parser(self):
    #     self.config['algorithm'] = self.algorithm_name
    #     self.config['random_baseline'] = self.explains['random_baseline']
    #     self.config['random_iter'] = self.explains['random_iter']
    #     self.config['gradient_step'] = self.explains['gradient_step']
        
    # def _lime_parser(self):
    #     self.config['algorithm'] = self.algorithm_name
    #     self.config['segments'] = self.explains['segments']
    #     self.config['seed'] = self.explains['seed']
    #     self.config['num_samples'] = self.explains['num_samples']
    #     self.config['num_features'] = self.explains['num_features']
    #     self.config['positive_only'] = self.explains['positive_only']
    #     self.config['hide_rest'] = self.explains['hide_rest']
        
    # def _kernelshap_parser(self):
    #     self.config['algorithm'] = self.algorithm_name
    #     self.config['segments'] = self.explains['segments']
    #     self.config['seed'] = self.explains['seed']
    #     self.config['nsamples'] = self.explains['nsamples']
    
    def _public_check_config(self):
        frameworks = ['torch', 'darknet','dtrain']
        projects = ['train','explain','evaluation']
        if self.project_type not in [project.lower() for project in projects]:
            msg = f"The type you entered is:'{self.project_type}' Supported types are: {projects}"
            raise Exception(msg)
        elif not isinstance(self.data_resize, (tuple,list)):
            raise Exception("Supported types are: 'tuple' or 'list'")
        elif self.framework not in [framework.lower() for framework in frameworks]:
            msg = f"The type you entered is:'{self.framework}' Supported types are: {frameworks}"
            raise Exception(msg)
        
    def _explain_check_config(self):
        attributions = ['GradCAM', 'GradCAMpp', 'EigenCAM',"GuidedGradCAM", 'LRP', 'LRPYolo', 'IG', 'Lime','Kernelshap', 'Gradient', 'GradientxInput', 'Smoothgrad', 'DeepLIFT']
        if self.algorithm_name not in [attribution.lower() for attribution in attributions]:
            msg = f"The type you entered is:'{self.algorithm_name}' Supported types are: {attributions}"
            raise Exception(msg)
        else:
            if self.algorithm_name == 'gradcam':
                self.algorithm = GradCAM
            elif self.algorithm_name == 'gradcampp':
                self.algorithm = GradCAMpp
            elif self.algorithm_name == "guidedgradcam":
                self.algorithm = GuidedGradCAM
            elif self.algorithm_name == 'eigencam':
                self.algorithm = EigenCAM
            elif self.algorithm_name == 'lrp':
                self.algorithm = LRP
            elif self.algorithm_name == 'lrpyolo':
                self.algorithm = LRPYolo
            elif self.algorithm_name == 'ig':
                self.algorithm = IG
            elif self.algorithm_name == "lime":
                self.algorithm = LimeImage
            elif self.algorithm_name == "kernelshap":
                self.algorithm = KernelShap
            elif self.algorithm_name == 'gradient':
                self.algorithm = Gradient
            elif self.algorithm_name == 'gradientxinput':
                self.algorithm = GradientxInput
            elif self.algorithm_name == 'smoothgrad':
                self.algorithm = SmoothGrad
            elif self.algorithm_name == 'deeplift':
                self.algorithm = DeepLIFT
            
        
    def _evaluation_check_config(self):
        if self.config['metric'] == 'abpc':
            self.metric = AbPC
        elif self.config['metric'] == 'sensitivity':
            self.metric = Sensitivity
    def _train_check_config(self):
        improvement_algorithms = ['ABN', 'DomainGeneralization', 'DANN', 'DANN_GRAD', 'Default','FGSM']
        optimizers = ['Adam', 'SGD']
        loss_functions = ['CrossEntropyLoss', 'NLLLoss', 'MSELoss']
        if os.path.isdir(self.save_path) == False:
                os.makedirs(self.save_path) 
        # paser check                
        if self.algorithm_name not in [improvement_algorithm.lower() for improvement_algorithm in improvement_algorithms]:
            msg = f"The type you entered is:'{self.improvement_algorithm}' Supported types are: {improvement_algorithms}"
            raise Exception(msg)
        elif self.optimizer not in [optimizer.lower() for optimizer in optimizers]:
            msg = f"The type you entered is:'{self.optimizer}' Supported types are: {optimizers}"
            raise Exception(msg)
        elif self.loss_function not in [loss_function.lower() for loss_function in loss_functions]:
            msg = f"The type you entered is:'{self.loss_function}' Supported types are: {loss_functions}"
            raise Exception(msg)
        # optimizer
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam
        elif self.optimizer == 'sgd':
            self.optimizer = optim.SGD
        # loss function
        if self.loss_function == 'crossentropyloss':
            self.loss_function = nn.CrossEntropyLoss
        elif self.loss_function == 'nllloss':
            self.loss_function = F.nll_loss
        elif self.loss_function == 'mseloss':
            self.loss_function = nn.MSELoss
        # improvement algorithm
        
        if self.algorithm_name == 'abn':
            self.model_algorithm = self.algorithm_name
            self.improvement_algorithm = ABN
            # self.improvement_algorithm.name = 'abn'
            self._make_abn_model()
        elif self.algorithm_name == 'domaingeneralization':
            self.model_algorithm = self.algorithm_name
            self.set_freq = self.improvement_algorithm['set_freq']
            self.target_layer = self.improvement_algorithm['target_layer']
            self.improvement_algorithm = DomainGeneralization
            # self.improvement_algorithm.name = 'dg'            
            self._make_model()
        elif self.algorithm_name == 'default':
            self.model_algorithm = self.algorithm_name
            self.improvement_algorithm = Trainer
            # self.improvement_algorithm.name = 'default'
            self._make_model()
        elif self.algorithm_name == 'fgsm':
            epsilon = self.improvement_algorithm['epsilon']
            alpha = self.improvement_algorithm['alpha']
            self.model_algorithm = self.algorithm_name
            self.improvement_algorithm = FGSM
            self.improvement_algorithm.epsilon = epsilon
            self.improvement_algorithm.alpha = alpha
            self._make_model()
        elif self.algorithm_name == 'dann':
            self.model_algorithm = self.algorithm_name
            self.improvement_algorithm = DANN
            self._make_dann_model()
        elif self.algorithm_name == 'dann_grad':
            self.model_algorithm = self.algorithm_name
            self.target_layer = self.improvement_algorithm['target_layer']
            self.improvement_algorithm = DANN_GRAD
            self.improvement_algorithm.target_layer = self.target_layer
            self._make_dann_model()
            
        if self.gpu_count >0:
            gpus = []
            for i in range(self.gpu_count):
                gpus.append(i)
            self.gpu_count = gpus
        else:
            msg = f"The value you entered is:'{self.gpu_count}' The value must be greater than or equal to 1."
            raise Exception(msg)
        
    def _make_model(self):
        if self.model_name.startswith("resnet"):
            self.make_model = models.resnet50
        elif self.model_name.startswith("vgg"):
            self.make_model = models.vgg19
         
    def _make_abn_model(self):
        if self.model_name.startswith("resnet"):
            self.make_model = make_attention_resnet50
        elif self.model_name.startswith("vgg"):
            self.make_model = make_attention_vgg19
            
    def _make_dann_model(self):
        if self.model_name.startswith("resnet"):
            self.make_model = make_dann_resnet50
        elif self.model_name.startswith("vgg"):
            raise Exception("Not Supported")
            # self.make_model = models.vgg19
