from KonanXAI.attribution.layer_wise_propagation.lrp_tracer import Graph
from ...utils import *

## test
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import copy
__all__ = ["LRP"]
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
        self.label_index = None
        self.rule = config['rule']
        self.alpha = None
        self.yaml_path = None
        if "alpha" in config:
            self.alpha = config["alpha"]
        else:
            self.alpha = sys.float_info.epsilon

    def calculate(self, inputs=None, targets=None):
        if inputs != None:
            self.input = inputs
        if targets != None:
            self.label_index = targets
        model =  copy.deepcopy(self.model)
        model.eval()
        self.tracer =  Graph(model, self.yaml_path)
        self.module_tree = self.tracer.trace()
        if model.model_algorithm == 'abn':
            att, pred, _ = model(self.input).squeeze()
        else:
           pred = model(self.input).squeeze()
        # pred = model(self.input).squeeze()
        if self.label_index == None:
            self.label_index = pred.argmax()
            max = pred.max()
        else:
            max = pred[self.label_index]
        relevance = torch.zeros([model.output_size], device = 'cuda:0')
        relevance[self.label_index] = max
        relevance = self.module_tree.backprop(relevance, self.rule, self.alpha)
        
        relevance = torch.sum(relevance, axis=1)
        relevance = relevance.unsqueeze(1)
        return relevance.cpu().detach()