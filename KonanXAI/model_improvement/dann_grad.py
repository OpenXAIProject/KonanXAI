import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dann import DANN
from tqdm import tqdm

class DANN_GRAD(DANN):       
    def _get_target_layer(self):
        if isinstance(self.target_layer, list):
            self.layer = self.model._modules[self.target_layer[0]]
            for layer in self.target_layer[1:]:
                self.layer = self.layer._modules[layer]
        self.layer.fwd_y = []
    def _fwd_hook(l, x, y):
        l.fwd_y.append(y[0])
    
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
            filename = f"DANN_GRAD_{model_class}_{dataset_class}_{epoch}ep.pt"
        else:
            filename = f"DANN_GRAD_{model_class}_{dataset_class}_{epoch}ep_final.pt"
            d['accuracy'] = accuracy
        msg = f"[CHECKPOINT] '{self.save_path}/{filename}' save done."
        torch.save(d, self.save_path + "/" + filename)
        print(msg)
    
    def _bwd_hook(l, xg, yg):
        if len(l.fwd_y) > 2: #resnet
            if xg[0].shape[1] == 2048:
                def norm(x):
                    min_x = torch.min(x)
                    max_x = torch.max(x)
                    n = (x - min_x) / (min_x + max_x)
                    return 2 * n - 1
                M = l.fwd_y[2] + l.fwd_y[5]
                M = norm(M)
                grad = xg[0] + xg[0] * M * l.alpha
                return (grad, )
        else: #vgg
            def norm(x):
                min_x = torch.min(x)
                max_x = torch.max(x)
                n = (x - min_x) / (min_x + max_x)
                return 2 * n - 1
            M = l.fwd_y[0] + l.fwd_y[1]
            M = norm(M)
            grad = xg[0] + xg[0] * M * l.alpha
            return (grad, )
    
    def _clear_target_layer(self):
        self.layer.fwd_y = []
        self.layer_x = []
        
    def _set_target_layer_hook(self):
        self.fwd_handle = self.layer.register_forward_hook(DANN_GRAD._fwd_hook)
        self.bwd_handle = self.layer.register_full_backward_hook(DANN_GRAD._bwd_hook)
        
    def _hook_pre_forward(self, x, y, epoch, i):
        super()._hook_pre_forward(x, y, epoch, i)
        self.layer.alpha = self.alpha
        
    def _train_init(self):
        super()._train_init()
        self._get_target_layer()
        self._clear_target_layer()
        self._set_target_layer_hook()
    
    def _hook_next_update(self, x_batch, y_batch, pred, loss, epoch):
        self._clear_target_layer()