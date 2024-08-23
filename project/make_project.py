import os
from KonanXAI.utils.heatmap import compose_heatmap_image, get_heatmap
from project.config import Configuration
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset
from KonanXAI.attribution.layer_wise_propagation.lrp import LRP
from KonanXAI.attribution.layer_wise_propagation.lrp_yolo import LRPYolo
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
        heatmaps = []

        for i, data in enumerate(self.dataset):
            if self.framework.lower() == 'darknet':
                origin_img = data.origin_img
                img_size = data.im_size
            else:
                origin_img = data[0]
                img_size = data[3]
            algorithm_type = self.config['algorithm']
            img_path = self.dataset.train_items[i][0].split('\\')
            root = f"{self.save_path}{self.algorithm_name}_result/{img_path[-2]}"
            if os.path.isdir(root) == False:
                os.makedirs(root)
            img_save_path = f"{root}/{img_path[-1]}"
            algorithm = globals().get(self.algorithm_name)(self.framework, self.model, data, self.config)
            
            heatmap = algorithm.calculate()
            get_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, self.framework)
            
            # 추후 사용될 수 있음(clustering)
            # heatmaps.append(heatmap)
    
                