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
        self._explain_algorithm_parser()
        
    
    
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
        self.algorithm_name = list(self.explain_algorithm[0].keys())[0]
        
    def _explain_algorithm_parser(self):
        if self.algorithm_name == 'GradCAM':
            self.config = self._gradcam_parser()
            return self.config
        
        elif self.algorithm_name == 'LRP':
            self.config = self._lrp_parser()
            return self.config
        
    def _gradcam_parser(self):
        self.target_layer = list(self.explain_algorithm[0].items())[0][1][0]['target_layer']
        return (self.target_layer,)
    
    def _lrp_parser(self):
        self.rule = list(self.explain_algorithm[0].items())[0][1][0]['rule']
        return (self.rule,)
