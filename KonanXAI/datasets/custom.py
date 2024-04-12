from .manager import Datasets, darknet
import os
class Custom(Datasets):
    def __init__(self):
        super().__init__()
        dir_list = len(os.listdir("D:/xai_refactoring/data"))
        self.data = [f"data/{i+1}.jpg" for i in range(dir_list)]
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = darknet.open_image(self.data[idx], (608, 608))#(640, 640))
        return image