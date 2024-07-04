from abc import *
from ...utils.enum import *
from ..kernel import ExplainData
from ...models import XAIModel
from ...datasets import Datasets

class Algorithm:
    def __init__(self, model: XAIModel, dataset: Datasets, platform: PlatformType):
        self.model = model
        self.dataset = dataset
        self.platform = platform
        self.dataset_convert()

    def dataset_convert(self):
        if self.platform == PlatformType.Darknet:
            self.dataset.toDarknet()
        elif self.platform == PlatformType.DTrain:
            self.dataset.toDtrain()
        elif self.platform == PlatformType.Pytorch:
            self.dataset.toTensor()

    @abstractmethod
    def calculate(self):
        raise NotImplementedError