import os
from KonanXAI.utils.data_convert import convert_tensor
from KonanXAI.utils.evaluation import ZeroBaselineFunction, heatmap_postprocessing, postprocessed_guided, postprocessed_ig
from KonanXAI.utils.heatmap import get_heatmap, get_kernelshap_image, get_lime_image, get_scale_heatmap, get_guided_heatmap, get_ig_heatmap
from project.config import Configuration
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def record_evaluation(self, dict_origin, image_path, metric_type, metric_li, attr_tpye, score):
        metric_li["image_path"] = image_path
        metric_li[attr_tpye] = score
        return metric_li
    
    def postprocessing_eval(self, result_json, metrics, metric_name):
        metric = metric_name
        score_li = []
        for value_dict in result_json[metric]:
            score_li.append(list(value_dict.values())[1:])
            keys = list(value_dict.keys())[1:]
        score_average = [list(map(lambda x: sum(x) / len(score_li), zip(*score_li)))]
        metric_dict = {}
        metric_name = f"result_{metric}"
        result_json[metric_name] = []
        for key, value in zip(keys, score_average[0]):
            metric_dict[key] = round(value,4)
        metric_dict = dict(sorted(metric_dict.items(), key=lambda item: item[1], reverse=True))
        result_json[metric_name].append(metric_dict)
        return result_json

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
        print(f"improvement_algorithm: {trainer}")
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

    def explain(self):
        for i, data in enumerate(self.dataset):
            origin_img, img_size, output = self.preprocessing(data)
            algorithm_type = self.config['algorithm']
            if self.framework == 'dtrain':
                img_path = ["data/", self.dataset.image_name[i]]
            else:
                img_path = self.dataset.test_items[i][0].split('\\')
            root = f"{self.save_path}{self.algorithm_name}_result/{img_path[-2]}"
            if os.path.isdir(root) == False:
                os.makedirs(root)
            img_save_path = f"{root}/{img_path[-1]}"
            print(self.algorithm)
            algorithm = self.algorithm(self.framework, self.model, data, self.config)
            algorithm.data_type = self.dataset.dataset_name
            heatmap = algorithm.calculate(targets=output)
                
            if "eigencam" in self.algorithm_name and 'yolo' in self.model.model_name:
                if 'af_yolo' in self.model_name:
                    get_scale_heatmap(origin_img, heatmap, img_save_path, img_size,algorithm_type, self.framework, reverse=True)
                else:
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

    def explainer(self):
        algorithm = self.algorithm(self.framework, self.model, self.dataset, self.config)
        algorithm.data_type = self.dataset.dataset_name
        # counterfacual도 evaluation이 있을텐데?
        algorithm.apply()


    def eval(self):
        set_seed(777)
        evaluation_result={}
        evaluation_result[self.config['metric']] = []
        for i, data in enumerate(self.dataset):
            metric_li = {}
            print(f"Evaluation: {i+1}/{len(self.dataset)}")
            origin_img, img_size, output = self.preprocessing(data)
            algorithm = self.algorithm(self.framework, self.model, data, self.config)
            algorithm.type = self.algorithm_name
            algorithm.data_type = self.dataset.dataset_name
            heatmap = algorithm.calculate()
            heatmap = heatmap_postprocessing(self.algorithm_name, img_size, heatmap)
            print(f"evaluation Metric: {self.metric}")
            if self.config['metric'] == 'abpc':                 
                score = round(self.metric(model=self.model, baseline_fn=ZeroBaselineFunction()).evaluate(inputs=(data[0].to('cuda')),targets=output, attributions=heatmap).item(),4)#.squeeze(0))
                self.record_evaluation(evaluation_result,self.dataset.image_name[i], self.config['metric'], metric_li, algorithm.type, score)
            elif self.config['metric'] == 'sensitivity':
                score = round(self.metric(model = self.model).evaluate(inputs= origin_img.to(self.device),targets=output, attributions=heatmap, explainer = algorithm).item(),4)
                self.record_evaluation(evaluation_result,self.dataset.image_name[i], self.config['metric'], metric_li, algorithm.type, score)
            evaluation_result[self.config['metric']].append(metric_li)
            print(f"data_path:{self.dataset.image_name[i]}\nscore: {score}")
        with open(f"{self.save_path}/{self.config['metric']}_resutl.json", "w") as f:
            json.dump(self.postprocessing_eval(evaluation_result, self.metric, self.config['metric']),f, indent=4)
            
    def preprocessing(self, data):
        if self.framework == 'darknet':
            origin_img = data.origin_img
            img_size = data.im_size
            output = None
        else:
            img_size = data[3]
            origin_img = convert_tensor(data[4], "origin", img_size).unsqueeze(0)
            if data[1] == -1:
                if "yolo" not in self.model_name:
                    if self.dataset.dataset_name == "imagenet":
                        infer_data = convert_tensor(data[4], self.dataset.dataset_name, img_size).unsqueeze(0).to(self.device)
                    else:
                        infer_data = data[0]
                    output = self.model(infer_data).argmax(-1).item()
                else:
                    output = None
            else: 
                output = data[1]
        return origin_img, img_size, output
            
    def run(self):
        self.dataset = load_dataset(self.framework, data_path = self.data_path,
                                    data_type = self.data_type, resize = self.data_resize, mode = self.project_type)
        self.model = model_import(self.framework, self.source, self.repo_or_dir,
                                  self.model_name, self.cache_or_local, 
                                  self.weight_path, self.cfg_path, self.dataset.classes, self.model_algorithm)
        print(self.model)
        
        if self.project_type == "explain":
            self.explain()
        elif self.project_type == "train":
            self.train()
        elif self.project_type == "explainer":
            self.explainer()
        elif self.project_type == 'evaluation':
            self.eval()