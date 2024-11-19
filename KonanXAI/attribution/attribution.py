from abc import *
#from ...utils.enum import *
#from ..kernel import ExplainData
#from ...models import XAIModel
from KonanXAI.datasets import Datasets
__all__ = []
class Attribution:
    def __init__(self, framework, model, dataset: Datasets):
        self.framework = framework
        self.model = model
        self.dataset = dataset
        self.dataset_convert()

    def dataset_convert(self):
        if self.framework == 'darknet':
            self.dataset.toDarknet()
        elif self.framework == 'dtrain':
            self.dataset.toDtrain()
        elif self.framework == 'torch':
            self.dataset.toTensor()

    @abstractmethod
    def calculate(self):
        raise NotImplementedError