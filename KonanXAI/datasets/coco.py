from . import Datasets
import os
from glob import glob
class COCO(Datasets):
    def __init__(self, framework, src_path):
        super().__init__(framework, src_path)
        self.framework = framework
        self.src_path = src_path
        self.dataset_name = 'coco'
        self.classes = 80
        self.make_cache = {}
        
    def load_src_path(self):
        train_path = glob(self.src_path+"/images/train2017/*.*")
        test_path = glob(self.src_path+"/images/test2017/*.*")# 수정 필요
        self.train_items = []
        self.val_items = []
        self.test_items = []
        for path in train_path:
            label_path = path.replace("images","labels").replace(".jpg",".txt")
            labels = []
            try:
                with open(label_path,"r") as file:
                    data = file.readlines()
                for item in data:
                    item = item.replace("\n","").split(" ")
                    label = int(item[0])
                    bbox = [float(i) for i in item[1:5]]
                    bbox.insert(0,label)
                    labels.append(bbox)
                self.train_items.append((path, labels))
            except:
                pass
            if len(self.train_items) == 10:
                break

        for path in test_path:
            # label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
            # label = int(label) - 1
            label = None
            self.test_items.append((path, label))