from project.manager import Configuration
"""
2024-07-02 jjh
 
usage:
    project = make_attribution()
    projcet.run()
"""
class make_attirbution(Configuration):
    def __init__(self, config_path:str):
        super().__init__(config_path)
        # model
        self.mtype = self._check_model_name()
        # platform
        self.platform = self._check_platform_type()
        # dataset
        self.dtype = self._check_data_type()
        # explain
        self.etype = self._check_explain_type()
    
    def set_xai(self):      
        self.xai.load_model_support(self.mtype, self.platform, weight_path = self.weight_path)
        self.xai.load_dataset_support(self.dtype, maxlen=10, path = self.data_path, fit_size = self.data_resize)
        self.xai.set_explain_mode([self.etype])
        
    def run(self):
        self.set_xai()
        self.get_target_layer(self.xai.model.net)
        explain = self.xai.explain(target_layer = self.target_layer)
        explain.save_heatmap(self.save_path)