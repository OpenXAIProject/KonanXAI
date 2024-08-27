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
        self._check_config()
        
    def _check_config(self):
        attributions = ['GradCAM', 'GradCAMpp', 'EigenCAM', 'LRP', 'LRPYolo']
        frameworks = ['torch', 'darknet']
        projects = ['explain', 'train']
        if self.project_type.lower() not in [project.lower() for project in projects]:
            msg = f"The type you entered is:'{self.project_type}' Supported types are: {projects}"
            raise Exception(msg)
        elif not isinstance(self.data_resize, (tuple,list)):
            raise Exception("Supported types are: 'tuple' or 'list'")
        elif self.framework.lower() not in [framework.lower() for framework in frameworks]:
            msg = f"The type you entered is:'{self.framework}' Supported types are: {frameworks}"
            raise Exception(msg)
        elif self.algorithm_name.lower() not in [attribution.lower() for attribution in attributions]:
            msg = f"The type you entered is:'{self.algorithm_name}' Supported types are: {attributions}"
            raise Exception(msg)
        
    def _parser_config(self):
        self.save_path = self.config['project']['save_path']
        self.weight_path = self.config['project']['weight_path']
        self.project_type = self.config['project']['project_type']
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