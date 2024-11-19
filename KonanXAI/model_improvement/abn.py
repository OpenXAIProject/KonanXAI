import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .trainer import Trainer
from tqdm import tqdm
__all__ = ["ABN"]
class ABN(Trainer):
    def __init__(self, model, optimizer, criterion, datasets, lr, batch, epoch, save_path):
        super(ABN, self).__init__(model, optimizer, criterion, datasets, lr, batch, epoch, save_path)
     
    def model_load(self, pt_path):
        pt = torch.load(pt_path)
        model_key = next(iter(self.model.state_dict()))
        state_dict = {}
        if 'module.' in model_key:
            for k, v in pt['model_state_dict'].items():
                key = 'module.'+ k 
                state_dict[key] = v
        else:
            for k, v in pt['model_state_dict'].items():
                key = k[7:] if k.startswith('module.') else k
                state_dict[key] = v
        self.model.load_state_dict(state_dict, strict = False)
        
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
            filename = f"ABN_{model_class}_{dataset_class}_{epoch}ep.pt"
        else:
            filename = f"ABN_{model_class}_{dataset_class}_{epoch}ep_final.pt"
            d['accuracy'] = accuracy
        msg = f"[CHECKPOINT] '{self.save_path}/{filename}' save done."
        torch.save(d, self.save_path + "/" + filename)
        print(msg)
        
    def _loss(self, pred, y):
        att, out, _ = pred
        att_loss = self.criterion(att, y)
        out_loss = self.criterion(out, y)
        return att_loss+out_loss#self.criterion(pred, y)
    
        
    def test(self, save=True):
        self.model.eval()
        self.datasets.set_test()
        self.datasets.set_batch(1)
        self.datasets.set_fit_size()
        acc = 0
        top5_error = 0
        for (x_batch, y_batch, custom, _) in tqdm(self.datasets):
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)
            att, pred, _ = self.model(x)
            top5 = torch.topk(pred, 5).indices.cpu().numpy()[0]
            pred = torch.argmax(pred).item()
            y = y.item()#torch.argmax(y).item()
            if pred == y:
                acc += 1
            if y in top5:
                top5_error += 1
            # print("Y :", y)
            # print("Pred :", pred)
            # print("Top5 :", top5)
        acc = round(acc / len(self.datasets) * 100, 2)
        top5_error = round(top5_error / len(self.datasets) * 100, 2)
        print(f"[TEST] Top1 Accuracy : {acc}%")
        print(f"[TEST] Top5 Accuracy : {top5_error}%")
        if save:
            self.model_save(self.epoch, acc)