from KonanXAI.lib.attribution.lrp.lrp_tracer import Graph
from ....models import XAIModel
from ....datasets import Datasets
from ..algorithm import Algorithm
from ....utils import *

## test
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import copy
class LRP(Algorithm): 
    def __init__(self, model: XAIModel, dataset: Datasets, platform: PlatformType):
        super().__init__(model, dataset, platform)
        self.rule = self.model.rule
        self.alpha = None
        
        if self.alpha != 'None':
            try:
                self.alpha = int(self.alpha)
            except: #epsilon rule default = 1e-8
                self.alpha = sys.float_info.epsilon

    def calculate(self):
        model =  copy.deepcopy(self.model.net)
        model.eval()
        self.tracer =  Graph(model, None)
        self.module_tree = self.tracer.trace()
        pred = model(self.target_input).squeeze()
        
        index = pred.argmax()
        
        relevance = torch.zeros([1000], device = 'cuda:0')
        max = pred.max()
        relevance[index] = max
        relevance = self.module_tree.backprop(relevance, self.rule, self.alpha)
        
        relevance = torch.sum(relevance, axis=1)
        relevance = relevance.unsqueeze(1)
        return relevance.cpu().detach()