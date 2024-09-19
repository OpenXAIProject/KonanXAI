import os
from KonanXAI.utils.heatmap import get_heatmap, get_lime_image, get_scale_heatmap, get_guided_heatmap, get_ig_heatmap
from project.config import Configuration
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset
import random
import numpy as np
import torch
import torch.nn as nn
"""
2024-07-02 jjh
 
usage:
    project = Project(config_path)
    projcet.run()
"""
def set_seed(seed_value=77):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) 
    

class Project(Configuration):
    def __init__(self, config_path:str):
        Configuration.__init__(self, config_path)
    
    def train(self):
        set_seed(777)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        model = self.make_model(num_classes= self.dataset.classes)
        for name, module in model.named_modules():
            if isinstance(module, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
                module.inplace = False  

        optimizer = self.optimizer(model.parameters(), lr = self.learning_rate)
        criterion = self.loss_function()
        trainer = self.improvement_algorithm(model, optimizer, criterion, self.dataset, self.learning_rate,self.batch_size, self.epoch, self.save_path)
        trainer.set_device(self.gpu_count)
        trainer.set_checkpoint_step(self.save_step)
        if self.transfer_weights != None:
            trainer.model_load(self.transfer_weights)
        if self.model_algorithm == 'domaingeneralization':
            trainer.set_freq(self.set_freq)
            target = model
            for m in self.target_layer:
                if m.isdigit():
                    target = target[int(m)]
                else:
                    target = getattr(target,m)
            trainer.set_target_layer(target)
        trainer.run()
        print("end")
    def explain(self):
        for i, data in enumerate(self.dataset):
            if self.framework == 'darknet':
                origin_img = data.origin_img
                img_size = data.im_size
            else:
                origin_img = data[0]
                img_size = data[3]
            algorithm_type = self.config['algorithm']
            img_path = self.dataset.test_items[i][0].split('\\')
            root = f"{self.save_path}{self.algorithm_name}_result/{img_path[-2]}"
            if os.path.isdir(root) == False:
                os.makedirs(root)
            img_save_path = f"{root}/{img_path[-1]}"
            algorithm = self.algorithm(self.framework, self.model, data, self.config)
            heatmap = algorithm.calculate()
                
            if "eigencam" in self.algorithm_name and 'yolo' in self.model.model_name:
                get_scale_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, self.framework)
            elif "guided" in self.algorithm_name:
                get_guided_heatmap(heatmap, img_save_path, img_size,algorithm_type, self.framework)
            elif "ig" == self.algorithm_name:
                get_ig_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, self.framework)
            elif "lime" == self.algorithm_name:
                get_lime_image(heatmap, img_save_path)
            else:
                get_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, self.framework)
            
    def run(self):
        self.dataset = load_dataset(self.framework, data_path = self.data_path,
                                    data_type = self.data_type, resize = self.data_resize, mode = self.project_type)
        self.model = model_import(self.framework, self.source, self.repo_or_dir,
                                  self.model_name, self.cache_or_local, 
                                  self.weight_path, self.cfg_path, self.dataset.classes, self.model_algorithm)
        
        
        if self.project_type == "explain":
            self.explain()
        elif self.project_type == "train":
            self.train()
            
                