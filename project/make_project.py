import os
from KonanXAI.utils.heatmap import compose_heatmap_image, get_heatmap
from project.config import Configuration
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset
import random
import numpy as np
import torch
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
            if self.framework.lower() == 'darknet':
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
            
                