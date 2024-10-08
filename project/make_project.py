import os
from KonanXAI.utils.evaluation import ZeroBaselineFunction
from KonanXAI.utils.heatmap import get_heatmap, get_kernelshap_image, get_lime_image, get_scale_heatmap, get_guided_heatmap, get_ig_heatmap, linear_transform, normalize_heatmap
from project.config import Configuration
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
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
            if self.framework.lower() == 'darknet':
                origin_img = data.origin_img
                img_size = data.im_size
            else:
                origin_img = data[0]
                img_size = data[3]
            algorithm_type = self.config['algorithm']
            if self.framework == 'dtrain':
                img_path = ["data/", self.dataset.image_name[i]]
            else:
                img_path = self.dataset.image_name[i].split('\\')
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
            elif "kernelshap" == self.algorithm_name:
                get_kernelshap_image(origin_img, heatmap, img_save_path, self.framework)
            else:
                get_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, self.framework)
    
    def eval(self):
        for i, data in enumerate(self.dataset):
            if self.framework == 'darknet':
                origin_img = data.origin_img
                img_size = data.im_size
            else:
                origin_img = data[0]
                img_size = data[3]

            output = self.model(data[0].to('cuda'))
            algorithm = self.algorithm(self.framework, self.model, data, self.config)
            heatmap = algorithm.calculate()
            
            if isinstance(heatmap, (tuple, list)):
                if "guided" in self.algorithm_name:
                    heatmap = postprocessed_guided((heatmap[0][0],heatmap[1][0][0]),1, img_size)
                else:
                    heatmap = heatmap[0]

            if self.algorithm_name == 'ig':
                heatmap = postprocessed_ig(heatmap,0)
            elif len(heatmap.shape) == 4:
                heatmap = F.interpolate(heatmap, img_size, mode='bilinear').detach().cpu().squeeze(0)
                             
            evaluation = self.metric(model=self.model, baseline_fn=ZeroBaselineFunction()).evaluate(inputs=(data[0].to('cuda')),targets=output.argmax(-1).item(), attributions=heatmap)#.squeeze(0))
            print(evaluation)
            
            
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
        elif self.project_type == 'evaluation':
            self.eval()
            
def postprocessed_ig(attr, dim):
    positive = np.clip(attr, 0, 1)
    gray_ig = np.average(positive, axis=2)
    linear_attr = linear_transform(gray_ig, 99, 0, 0.0, plot_distribution=False)
    ig_image = np.array(linear_attr, dtype=np.uint8)
    res = torch.tensor(ig_image)
    return res.clamp(min=0).sum(dim).unsqueeze(0)

def postprocessed_guided(attr, dim, img_size):
    heatmap, guided = attr
    heatmap = F.interpolate(heatmap, img_size, mode='bilinear').detach().cpu()
    heatmap_mask = np.transpose(heatmap.squeeze(0).cpu().numpy(),(1,2,0))
    res = heatmap_mask*guided
    res = torch.tensor(np.transpose(res,(2,0,1))).unsqueeze(0)
    poold = res.pow(2).sum(dim).sqrt()
    # poold = res.clamp(min=0).sum(dim)
    # norm = normalize_heatmap(poold)
    return poold

def deprocess_images(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return img
