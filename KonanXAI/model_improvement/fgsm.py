import cv2
import torch
import torch.nn.functional as F
import numpy as np
from .trainer import Trainer
import torch.nn as nn
from tqdm import tqdm
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
    
    def create_adversarial_attack(self, x, y, epsilon=8/255, alpha = 0.3):
        origin_x = x.requires_grad_(True)
        pred = self.model(origin_x)
        cost = self._loss(pred, y)
        self.model.zero_grad()
        cost.backward()
        adv_x = epsilon * origin_x.grad.detach().sign()
        return adv_x
    
    def _forward(self, x, y, c):
        delta = self.create_adversarial_attack(x, y, self.epsilon, self.alpha)
        return self.model(x + delta)
        
    # def test(self, save):
    #     self.model.eval()
    #     self.datasets.set_test()
    #     self.datasets.set_batch(2)
    #     self.datasets.set_fit_size()
    #     acc = 0
    #     top5_error = 0
    #     for (x_batch, y_batch, custom, _) in tqdm(self.datasets):
    #         x = x_batch.to(self.device)
    #         y = y_batch.to(self.device)
    #         delta = self.create_adversarial_attack(x, y, self.epsilon, self.alpha)
    #         pred = self.model(torch.clamp(x + delta, 0, 1))
    #         for pred_, y_ in zip(pred, y):
    #             pred_ = torch.argmax(pred_).item()
    #             y_ = y_.item()#torch.argmax(y).item()
    #             if pred_ == y_:
    #                 acc += 1
    #     acc = round(acc / (len(self.datasets)*self.datasets.batch) * 100, 2)
    #     top5_error = round(top5_error / len(self.datasets) * 100, 2)
    #     print(f"[TEST] Top1 Accuracy : {acc}%")
    #     # print(f"[TEST] Top5 Accuracy : {top5_error}%")
    #     if save:
    #         self.model_save(self.epoch, acc)
    