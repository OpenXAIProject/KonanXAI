import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .trainer import Trainer
from tqdm import tqdm
__all__ = ["DANN"]
class DANN(Trainer):           
    def model_save(self, epoch, accuracy=None):
        model_class = self.model.__class__.__name__
        dataset_class = self.datasets.__class__.__name__
        d = {
            'model_class': model_class,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.__class__.__name__,
            'dataset_class': dataset_class,
            'epoch': epoch,
            'batch': self.batch,
            'lr': self.lr,
            'loss_fn': self.criterion.__class__.__name__,
        }
        if accuracy is None: 
            filename = f"DANN_{model_class}_{dataset_class}_{epoch}ep.pt"
        else:
            filename = f"DANN_{model_class}_{dataset_class}_{epoch}ep_final.pt"
            d['accuracy'] = accuracy
        msg = f"[CHECKPOINT] '{self.save_path}/{filename}' save done."
        torch.save(d, self.save_path + "/" + filename)
        print(msg)
    
    def _set_parameters(self):
        for p in self.model.parameters():
            p.requires_grad = True
    
    def _set_device(self, x, y, c):
        c = c.to(self.device)
        return super()._set_device(x,y,c)
    
    def _train_init(self):
        super()._train_init()
        self._set_parameters()
        
    def _hook_pre_forward(self, x, y, epoch, i):
        p = float(i + epoch * len(self.datasets)) / (self.epoch * len(self.datasets))
        self.alpha = 2. / (1 + np.exp(-10 * p)) -1
        
    def _forward(self, x, y, c):
        self.model.zero_grad()
        source_label = torch.zeros(self.batch).long().to(self.device)
        Sp, Sd = self.model(x = x, alpha = self.alpha)
        
        target_label = torch.ones(self.batch).long().to(self.device)
        _, Td = self.model(x = c, alpha = self.alpha)
        return (Sp, Sd, Td, source_label, target_label)
    
    def _loss(self, pred, y):
        Sp, Sd, Td, source_label, target_label = pred
        Loss_1 = self.criterion(Sp, y)
        Loss_2 = self.criterion(Sd, source_label)
        Loss_3 = self.criterion(Td, target_label)
        L = Loss_1 + Loss_2 + Loss_3
        return L