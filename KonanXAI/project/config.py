import yaml
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import KonanXAI as XAI
class Configuration:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self._parser_config()
        print(self.explain_algorithm)
        
    
    
    def _parser_config(self):
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
        self.explain_algorithm = self.config['project']['explain_algorithm']
        

    # gradcam 의 method로 넣자
    def get_target_layer(self, model):
        if self.framework == "pytorch":
            self.target_layer = model
            target = self.config['config']['target_layer']
            if isinstance(target,list):
                for layer in target:
                    self.target_layer = self.target_layer._modules[layer]
            elif isinstance(target,dict):
                self.target_yolo_layer = []
                for index, layers in target.items():
                    base_layer = model
                    for layer in layers:
                        base_layer = base_layer._modules[layer]
                    self.target_yolo_layer.append(base_layer)
                self.target_layer = self.target_yolo_layer
            else:
                self.target_layer = None
                
        elif self.framework == "darknet":
            self.target_layer = None
        
   
