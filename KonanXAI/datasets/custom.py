from .manager import Datasets, darknet
import os
from glob import glob
class Custom(Datasets):
    def __init__(self):
        super().__init__()
        # dir_list = #os.listdir("D:/xai_refactoring/data")
        # self.data = [f"data/{i}.jpg" for i in dir_list]
        self.data = glob("./data/*.jpg")
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = darknet.open_image(self.data[idx], (640, 640))#(640, 640))
        return image