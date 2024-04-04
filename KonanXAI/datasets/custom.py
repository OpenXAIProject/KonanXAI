from .manager import Datasets, darknet

class Custom(Datasets):
    def __init__(self):
        super().__init__()
        self.data = ["asd.jpg"]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = darknet.open_image(self.data[idx], (640, 640))
        return image