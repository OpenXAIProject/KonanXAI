import cv2
import torch
import torch.nn.functional as F
import numpy as np
from .trainer import Trainer

class DomainGeneralization(Trainer):
    def _fwd_hook(l, x, y):
        if isinstance(l.fwd_x, dict):
            l.fwd_x[x[0].device] = x
            l.fwd_y[y[0].device] = y
        else:
            l.fwd_x = x
            l.fwd_y = y

    def _bwd_hook(l, xg, yg):
        l.bwd_xg = xg
        l.bwd_yg = yg
        
    def _fwd_mask_hook(l, x, y):
        if l.mask is not None:
            if isinstance(l.mask, dict):
                device = str(x[0].device)
                test = F.interpolate(y,size=(224,224), mode='bicubic', align_corners=False)
                res = l.mask[device] * test
                res = F.interpolate(y,size=(7,7), mode='bicubic', align_corners=False)
                return res
            else:
                return l.mask * y
        else:
            return y

    def set_freq(self, freq):
        self.freq = freq
        
    def set_target_layer(self, layer):
        self.target_layer = layer
        self.target_layer.register_forward_hook(DomainGeneralization._fwd_mask_hook)
        self.target_layer.mask = None
        
    def _hook_pre_forward(self, x_batch, y_batch, epoch):
        self.current_epoch = epoch + 1
        
    def _forward(self, x, y, c):
        self.target_layer.mask = None
        if self.current_epoch % self.freq == 0:
            saliency = self._get_saliency(x, y)
            saliency_resize = F.interpolate(saliency.unsqueeze(0), size=(224,224),mode="bicubic",align_corners=False).squeeze(0)
            # Batch save_shape
            # w, h = self._argmax_2d(saliency, save_shape=(0,))
            w, h = self._argmax_2d(saliency_resize, save_shape=(0, ))
            # Get feature size from selected layer
            # batch, fh, fw = saliency.size()
            resize_batch, resize_fh, resize_fw = saliency_resize.size()
            # Make Mask
            # gt_mask = self._get_mask(x, c, (fh, fw)).detach().cpu()
            gt_mask_resize = self._get_mask(x, c, (resize_fh, resize_fw)).detach().cpu()
            # indexes = (w + h * fw).unsqueeze(1)
            indexes = (w + h * resize_fw).unsqueeze(1)
            # condition = gt_mask.reshape((batch, -1)).gather(1, indexes).unsqueeze(1)
            condition = gt_mask_resize.reshape((resize_batch, -1)).gather(1, indexes).unsqueeze(1)
            s_mask = torch.where(saliency_resize > 0, True, False)
            # Check Condition
            mask = s_mask * condition + gt_mask_resize * ~condition
            mask = mask & gt_mask_resize
            # for index, value in enumerate(condition):
            #     if value == True:
            #         print("T:",index)
            mask = mask.unsqueeze(1)
            self.target_layer.mask = {}
            b = resize_batch // self.device_count
            for i in range(self.device_count):
                device = f'cuda:{i}'
                self.target_layer.mask[device] = mask[i*b:(i+1)*b].to(device)
            # self.target_layer.mask = mask.to(self.device)
            # Check mask
            return super()._forward(x, y, c)
        else:
            return super()._forward(x, y, c)
        

    def _get_saliency(self, x, y):
        fwd_handle = self.target_layer.register_forward_hook(DomainGeneralization._fwd_hook)
        if self.device_count > 0:
            self.target_layer.fwd_x = {}
            self.target_layer.fwd_y = {}
        x.requires_grad = True
        pred: torch.Tensor = self.model(x)
        # loss = self._loss(pred, y)
        index = y.unsqueeze(1)#y.argmax(1).unsqueeze(1)
        loss = pred.gather(dim=1, index=index).sum()
        # saliency (GradCAM)
        grads = tuple() 
        feats = tuple()
        for device, feat in self.target_layer.fwd_y.items():
            grads += (torch.autograd.grad(loss, feat, retain_graph=True)[0].detach().cpu(),)
            feats += (feat.detach().cpu(),)
        grad = torch.concat(grads, dim=0)
        feat = torch.concat(feats, dim=0)
        # grad = torch.autograd.grad(loss, feat, retain_graph=True)[0]
        # GAP
        ak = grad.mean((2, 3)).unsqueeze(2).unsqueeze(3)
        combination = (ak * feat).sum(1)
        saliency = F.relu(combination)
        # Clear
        self.model.zero_grad()
        self.optimizer.zero_grad()
        fwd_handle.remove()
        return saliency
    
    def _argmax_2d(self, tensor, save_shape, wd=None, hd=None):
        reshape = []
        for dim in save_shape:
            reshape.append(tensor.shape[dim])
        idx = tensor.reshape(tuple(reshape) + (-1, )).argmax(dim=len(save_shape))
        if wd is None or hd is None:
            hd = tensor.size(len(save_shape)-1)
            wd = tensor.size(len(save_shape))
        w, h = idx % wd, torch.div(idx, wd, rounding_mode= 'floor')#idx // wd
        return w, h
        
    def _get_mask(self, x_batch, mask, resize):
        xs = x_batch.cpu().detach().numpy()
        xs = np.uint8(xs * 255)
        if mask is None:
            mask_x = []
            # Loop Batch
            for x in xs:
                x = np.transpose(x, (1, 2, 0))
                img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                x = cv2.resize(thresh, dsize=resize, interpolation=cv2.INTER_CUBIC)
                mask_x.append(x)
            return torch.tensor(np.stack(mask_x), dtype=torch.bool).to(self.device)
        else:
            mask_x = []
            for m in mask:
                x = cv2.resize(m, dsize=resize, interpolation=cv2.INTER_CUBIC)
                mask_x.append(x)
            mask_x = np.stack(mask_x)
            mask_x = np.where(mask_x > 0, 1, 0)      
            return torch.tensor(mask_x, dtype=torch.bool).to(self.device)