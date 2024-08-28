import os
import torch
from KonanXAI.utils.heatmap import compose_heatmap_image, get_heatmap, heatmap_tensor
from KonanXAI.utils.h5file import create_attribution_database, create_dataset, append_attributions, append_sample
from project.config import Configuration
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset
from KonanXAI.attribution.layer_wise_propagation.lrp import LRP
from KonanXAI.attribution.layer_wise_propagation.lrp_yolo import LRPYolo
from KonanXAI.attribution.gradcam import GradCAM
from KonanXAI.explainer.clustering import SpectralClustering


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
        self.heatmaps = []

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
            algorithm = globals().get(self.algorithm_name)(self.framework, self.model, data, self.attr_config)
            
            heatmap = algorithm.calculate()
            get_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, self.framework)
            
            self.heatmaps.append(heatmap)

    def run_for_counterfactual(self):
        self.model = model_import(self.framework, self.source, self.repo_or_dir,
                                  self.model_name, self.cache_or_local, 
                                  self.weight_path, self.cfg_path)
        self.dataset = load_dataset(self.framework, data_path = self.data_path,
                                    data_type = self.data_type, resize = self.data_resize)
        print(self.dataset[0])


    def run_for_clustering(self):
        device = torch.device('cuda')
        self.model = model_import(self.framework, self.source, self.repo_or_dir,
                                  self.model_name, self.cache_or_local, 
                                  self.weight_path, self.cfg_path)
        self.dataset = load_dataset(self.framework, data_path = self.data_path,
                                    data_type = self.data_type, resize = self.data_resize)
        self.heatmaps = []

        number_of_dataset_processed = 0
        label = 0
        if os.path.isfile(self.exp_config['h5_dataset_file_path']) == False:
            with create_dataset(
                    self.exp_config['h5_dataset_file_path'],
                    self.dataset[0][0].shape,
                    len(self.dataset)
                ) as database_file:
                    for data in self.dataset:

                        append_sample(
                            database_file,
                            number_of_dataset_processed,
                            data[0],
                            label
                        )
                        number_of_dataset_processed += data[0].shape[0]
                        print(
                            f'Computed {number_of_dataset_processed}/{len(self.dataset)} dataset'
                        )
        number_of_attribution_processed = 0
        if os.path.isfile(self.exp_config['h5_attr_file_path']) == False:
            with create_attribution_database(
                self.exp_config['h5_attr_file_path'],
                self.dataset[0][0].shape,
                self.model.num_of_classes,
                len(self.dataset)
            ) as attribution_database_file:
                for i, data in enumerate(self.dataset):
                    if self.framework.lower() == 'darknet':
                        origin_img = data.origin_img
                        img_size = data.im_size
                    else:
                        origin_img = data[0]
                        img_size = data[3]
                    
                 

                    predictions = self.model(data[0].to(device))
                    predictions = predictions.detach().cpu()

                    algorithm_type = self.attr_config['algorithm']
                    
                    algorithm = globals().get(self.algorithm_name)(self.framework, self.model, data, self.attr_config)
                    
                    heatmap = algorithm.calculate()
                    heatmap = heatmap_tensor(origin_img, heatmap, img_size,algorithm_type, self.framework)

                    append_attributions(
                        attribution_database_file,
                        number_of_attribution_processed,
                        heatmap,
                        predictions,
                        torch.Tensor(label)
                    )
                    number_of_attribution_processed+=heatmap.shape[0]
                    print(
                        f'Computed {number_of_attribution_processed}/{len(self.dataset)} attributions'
                    )
             
            
        
    def clustering(self):
        self.model = model_import(self.framework, self.source, self.repo_or_dir,
                                  self.model_name, self.cache_or_local, 
                                  self.weight_path, self.cfg_path)
        self.dataset = load_dataset(self.framework, data_path = self.data_path,
                                    data_type = self.data_type, resize = self.data_resize)

        self.run_for_clustering()
        spectral_clustering = globals().get(self.explainer_name)(
            self.framework, self.model, self.dataset, self.exp_config)
        spectral_clustering.apply()
    
                