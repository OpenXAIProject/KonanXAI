from project.config import Configuration
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import Datasets, COCO, CIFAR10, MNIST, CUSTOM    


"""
2024-07-02 jjh
 
usage:
    project = make_attribution()
    projcet.run()
"""
class Project(Configuration):
    def __init__(self, config_path:str):
        Configuration().__init__(config_path)
       

        
    def run(self):
        self.model = model_import(self.framework, self.source, self.repo_or_dir,
                                  self.model_name, self.cache_or_local, 
                                  self.weight_path, self.cfg_path)
        self.dataset = 
        
        
        explain = self.xai.explain(target_layer = self.target_layer)
        explain.save_heatmap(self.save_path)