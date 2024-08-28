import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, datasets, lr, batch, epoch, save_path):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.datasets = datasets
        self.lr = lr
        self.batch = batch
        self.epoch = epoch
        self.save_path = save_path
        self.step = int(epoch * 0.2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
    def set_device(self, gpus:list):
        self.model = self.model.to(self.device)
        # DataParallel
        if torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            if self.device_count > 1:
                self.model = nn.DataParallel(self.model, device_ids= gpus)
        
    def set_checkpoint_step(self, step):
        self.step = step
        
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
            filename = f"Default_{model_class}_{dataset_class}_{epoch}ep.pt"
        else:
            filename = f"Default_{model_class}_{dataset_class}_{epoch}ep_final.pt"
            d['accuracy'] = accuracy
        msg = f"[CHECKPOINT] '{self.save_path}/{filename}' save done."
        torch.save(d, self.save_path + "/" + filename)
        print(msg)
        
    def model_load(self, pt_path):
        pt = torch.load(pt_path)
        state_dict = {}
        for k, v in pt['model_state_dict'].items():
            key = k[7:] if k.startswith('module.') else k
            state_dict[key] = v
        self.model.load_state_dict(state_dict)
        

    def _hook_pre_forward(self, x_batch, y_batch, epoch):
        pass
    
    def _hook_pre_backward(self, x_batch, y_batch, pred, loss, epoch):
        pass
    
    def _hook_next_update(self, x_batch, y_batch, pred, loss, epoch):
        pass
    

    def run(self, train=True, test=True, save=True):
        if train:
            self.train()
        if test:
            self.test(save)
            
    def _forward(self, x, y, c):
        return self.model(x)
    
    def _loss(self, pred, y):
        return self.criterion(pred, y)
    
    def _backward(self, loss):
        loss.backward()

    def _update(self):
        self.optimizer.step()
        
    def _set_device(self, x, y, c):
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y, c
        
    def train(self):
        self.model.train()
        self.datasets.set_train()
        self.datasets.set_batch(self.batch)
        self.datasets.set_fit_size()
        
        for epoch in range(self.epoch):
            self.datasets.shuffle()
            avg = 0
            print(f"[TRAIN] STEP {epoch + 1} / {self.epoch}")
            for (x_batch, y_batch, custom, _) in tqdm(self.datasets):
                x, y, c = self._set_device(x_batch, y_batch, custom)
                # x = x_batch.to(self.device)
                # y = y_batch.to(self.device)
                # Forward
                self._hook_pre_forward(x, y, epoch)
                pred = self._forward(x, y, c)
                loss = self._loss(pred, y)
                self._hook_pre_backward(x, y, pred, loss, epoch)
                # Backward
                self.optimizer.zero_grad()
                self._backward(loss)
                self._update()
                avg += loss.item()
                self._hook_next_update(x, y, pred, loss, epoch)
            avg /= len(self.datasets) * self.batch
            print(f"[TRAIN] {epoch + 1} EPOCH DONE. Avg.Loss : {avg}\n")

            if (epoch + 1) % 10 % self.step == 0:
                self.model_save(epoch + 1)
        
    def test(self, save):
        self.model.eval()
        self.datasets.set_test()
        self.datasets.set_batch(1)
        self.datasets.set_fit_size()
        acc = 0
        top5_error = 0
        for (x_batch, y_batch, custom, _) in tqdm(self.datasets):
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)
            pred = self.model(x)
            top5 = torch.topk(pred, 5).indices.cpu().numpy()[0]
            pred = torch.argmax(pred).item()
            y = y.item()#torch.argmax(y).item()
            if pred == y:
                acc += 1
            if y in top5:
                top5_error += 1
        acc = round(acc / len(self.datasets) * 100, 2)
        top5_error = round(top5_error / len(self.datasets) * 100, 2)
        print(f"[TEST] Top1 Accuracy : {acc}%")
        print(f"[TEST] Top5 Accuracy : {top5_error}%")
        if save:
            self.model_save(self.epoch, acc)