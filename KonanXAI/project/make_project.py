import os
from project.config import Configuration
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset, Datasets, COCO, CIFAR10, MNIST, CUSTOM    
from KonanXAI.attribution.gradcam import GradCAM

"""
2024-07-02 jjh
 
usage:
    project = Project(config_path)
    projcet.run()
"""
class Project(Configuration):
    def __init__(self, config_path:str):
        Configuration.__init__(self, config_path)
       

        
    def run(self):
        self.model = model_import(self.framework, self.source, self.repo_or_dir,
                                  self.model_name, self.cache_or_local, 
                                  self.weight_path, self.cfg_path)
        self.dataset = load_dataset(self.framework, data_path = self.data_path,
                                    data_type = self.data_type, resize = self.data_resize)
        
        print(self.explain_algorithm)
        self.target_layer = list(self.explain_algorithm[0].items())[0][1][0]['target_layer']
        algorithm_name = list(self.explain_algorithm[0].keys())[0]
        heatmaps = []

        for i, data in enumerate(self.dataset):
            
            print(1, self.dataset.train_items[i][0])
            img_path = self.dataset.train_items[i][0].split('\\')
            root = self.save_path + '{}_result\\'.format(algorithm_name) + img_path[-2]
            if os.path.isdir(root) == False:
                print(root)
                os.makedirs(root)
            

            img_save_path = root + '\\' + img_path[-1]
            algorithm = globals().get(algorithm_name)(self.framework, self.model, data, self.target_layer)
            
            heatmap = algorithm.calculate()
            heatmaps.append(heatmap)
            algorithm.get_heatmap(img_save_path)
    
                