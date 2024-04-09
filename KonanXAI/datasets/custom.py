from .manager import Datasets, darknet

class Custom(Datasets):
    def __init__(self):
        super().__init__()
        self.data = ["test.jpg"]#[f"{i%4+1}.jpg" for i in range(4)]
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = darknet.open_image(self.data[idx], (416, 416))#(640, 640))
        return image