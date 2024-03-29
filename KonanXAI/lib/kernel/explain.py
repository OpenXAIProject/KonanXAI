from abc import *

class ExplainData:
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass