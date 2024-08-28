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
        
        # self._explain_algorithm_parser()
        self._explainer_parser()
        
    
    
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
        if self.explain_algorithm!= None:
            self.algorithm_name = list(self.explain_algorithm[0].keys())[0]
        self.explainer = self.config['project']['explainer']
        self.explainer_name = list(self.explainer[0].keys())[0]


    def _explain_algorithm_parser(self):
        if self.algorithm_name == 'GradCAM':
            self.attr_config = self._gradcam_parser()
            return self.attr_config
        
        elif self.algorithm_name == 'LRP' or 'LRPYolo':
            self.attr_config = self._lrp_parser()
            return self.attr_config
        

        
    def _explainer_parser(self):
        if self.explainer_name == 'SpectralClustering':
            self.exp_config = self._clustering_parser()
            return self.exp_config
        elif self.explainer_name == 'Counterfactual':
            self.exp_config = self._counterfactual_parser()

        
    def _gradcam_parser(self):
        self.config = {}
        self.config['algorithm'] = self.algorithm_name
        self.config['target_layer'] = list(self.explain_algorithm[0].items())[0][1][0]['target_layer']
        return self.config
    
    def _lrp_parser(self):
        self.config = {}
        self.config['algorithm'] = self.algorithm_name
        self.config['rule'] = list(self.explain_algorithm[0].items())[0][1][0]['rule']
        self.config['yaml_path'] = self.cfg_path
        return self.config
    
    def _clustering_parser(self):
        self.config = {}
        self.config['explainer'] = self.explainer_name
        self.config['h5_dataset_file_path'] = list(self.explainer[0].items())[0][1][0]['h5_dataset_file_path']
        self.config['h5_attr_file_path'] = list(self.explainer[0].items())[0][1][1]['h5_attr_file_path']
        self.config['label_json_path'] = list(self.explainer[0].items())[0][1][2]['label_json_path']
        return self.config


    def _counterfactual_parser(self):
        self.config = {}
        self.config['explainer'] = self.explainer_name
        self.config['methods'] = list(self.explainer[0].items())[0][1][0]['methods']
        self.config['lambda'] = list(self.explainer[0].items())[0][1][1]['lambda']