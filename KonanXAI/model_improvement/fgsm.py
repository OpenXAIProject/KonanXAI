import cv2
import torch
import torch.nn.functional as F
import numpy as np
from .trainer import Trainer
import torch.nn as nn
class FGSM(Trainer):
    def model_save(self, epoch, accuracy=None):
        model_class = self.model.__class__.__name__
        dataset_class = self.datasets.__class__.__name__
        if isinstance(self.model,nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
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
            filename = f"FGSM_{model_class}_{dataset_class}_{epoch}ep.pt"
        else:
            filename = f"FGSM_{model_class}_{dataset_class}_{epoch}ep_final.pt"
            d['accuracy'] = accuracy
        msg = f"[CHECKPOINT] '{self.save_path}/{filename}' save done."
        torch.save(d, self.save_path + "/" + filename)
        print(msg)
    
    def create_adversarial_attack(self, x, y, epsilon=0.001, alpha = 0.3):
        delta = torch.zeros_like(x).uniform_(epsilon, epsilon).cuda()
        delta.requires_grad = True
        output = self.model(x + delta)
        loss = self._loss(output, y)
        self._backward(loss)
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), epsilon, epsilon)
        delta.data = torch.max(torch.min(1-x, delta.data), 0-x)
        delta = delta.detach()
        return delta
    
    def _forward(self, x, y, c):
        delta = self.create_adversarial_attack(x, y, self.epsilon, self.alpha)
        return self.model(torch.clamp(x + delta, 0, 1))
        
    