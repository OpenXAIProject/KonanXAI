from KonanXAI.attribution.layer_wise_propagation.lrp_tracer import Graph
from ...utils import *

## test
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import copy
class LRP: 
    def __init__(self, 
            framework, 
            model, 
            input, 
            config):
        # super().__init__(model, dataset, platform)
        self.framework = framework
        self.model = model
        self.model_name = self.model.model_name
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input = input[0].to(device)
        
        self.rule = config['rule']
        self.alpha = None
        self.yaml_path = None
        if self.alpha != 'None':
            try:
                self.alpha = int(self.alpha)
            except: #epsilon rule default = 1e-8
                self.alpha = sys.float_info.epsilon

    def calculate(self):
        model =  copy.deepcopy(self.model)
        model.eval()
        self.tracer =  Graph(model, self.yaml_path)
        self.module_tree = self.tracer.trace()
        pred = model(self.input).squeeze()
        
        index = pred.argmax()
        
        relevance = torch.zeros([1000], device = 'cuda:0')
        max = pred.max()
        relevance[index] = max
        relevance = self.module_tree.backprop(relevance, self.rule, self.alpha)
        
        relevance = torch.sum(relevance, axis=1)
        relevance = relevance.unsqueeze(1)
        return relevance.cpu().detach()