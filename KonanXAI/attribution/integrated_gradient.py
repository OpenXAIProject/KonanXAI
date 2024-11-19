from KonanXAI._core.pytorch.yolov5s.utils import non_max_suppression, yolo_choice_layer
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from copy import copy
__all__ = ["IG"]
class IG:
    def __init__(
            self, 
            framework, 
            model, 
            input, 
            config):
        '''
        input: [batch, channel, height, width] torch.Tensor 
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.framework = framework
        self.model = model
        self.model_name = self.model.model_name
        self.random_baseline = config['random_baseline']
        self.random_iter = config['random_iter']
        self.gradient_steps = config['gradient_step']
        self.label_index = None
        if framework == "darknet":
            self.input = input
            self.input_size = self.input.shape
        else:
            self.input = input[0].to(self.device)
            self.input_size = self.input.shape[2:4]
        
        
    def _preprocess(self, x):
        x = np.array(x)
        # if 'yolo' in self.model_name:
        mean = np.array([0., 0., 0.]).reshape([1, 1, 1, 3])
        std = np.array([1, 1, 1]).reshape([1, 1, 1, 3])
        # else:      
        # #ImageNet
        #     mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 1, 3])
        #     std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 1, 3])
        obs = x / 255.
        z_score = (obs - mean) / std
        obs = np.transpose(z_score, (0, 3, 1, 2))
        return torch.tensor(obs, dtype = torch.float32, requires_grad = True, device = self.device)    
    
    def calculate(self,inputs=None, targets= None):
        target_label = self.get_gradient(inputs, targets)
        if isinstance(target_label,int) or len(target_label)>0:
            iteration = self.random_iter if self.random_iter else 1
            igs = []
            for i in tqdm(range(iteration)):
                baseline = self._set_baseline()
                ig = self._integrated_gradients(baseline, target_label)
                if isinstance(ig, list):
                    igs_yolo = []
                    for v in ig:
                        igs_yolo.append(v)
                    igs.append(igs_yolo)
                else:
                    igs.append(ig)
            if isinstance(ig, list):
                heatmap = []
                igs = np.stack(igs,axis=1)
                for map in igs:
                    heatmap.append(np.average(np.array(map), axis=0))
            else:    
                heatmap = np.average(np.array(igs), axis=0)
            return heatmap       
        elif len(target_label) == 0:
            return
        
                        
    def get_gradient(self,inputs, targets):
        if inputs != None:
            self.input = inputs
        if targets != None:
            self.label_index = targets
        self.model = self.model.eval()
        x = self.input.requires_grad_().to(self.device)#self._preprocess([self.input])
        if isinstance(self.input, torch.Tensor):
            self.input = self.input.squeeze(0)
            self.input = self.input.detach().cpu().numpy()
            self.input = (self.input-np.min(self.input))/(np.max(self.input)-np.min(self.input))
            self.input = np.transpose(np.uint8(self.input * 255), (1,2,0))
        if "yolo" in self.model_name:
            target_label = []
            self.pred_origin, raw_logit = self.model(x)
            self.logits_origin = torch.concat([data.view(-1,self.pred_origin.shape[-1])[...,5:] for data in raw_logit],dim=0)
            with torch.no_grad():
                self.pred, self.logits, self.select_layers = non_max_suppression(self.pred_origin, self.logits_origin.unsqueeze(0), conf_thres=0.25, model_name = self.model_name)
            self.index_tmep = yolo_choice_layer(raw_logit, self.select_layers)
            for cls, sel_layer in zip(self.pred[0], self.select_layers):
                target_label.append([sel_layer, cls[5].item()])
        else:
            out = self.model(x)
            if self.label_index == None:
                target_label = torch.argmax(out.detach().cpu(), dim = 1).item()
            else:
                target_label = self.label_index
        return target_label
    
    def _set_baseline(self):
        if self.random_baseline:
            return 255 * np.random.random(self.input.shape)
        else:
            return np.zeros(self.input.shape, dtype = np.float32)
        
    def _integrated_gradients(self, base_line, target_label):
        steps = self.gradient_steps
        scale_inputs = [base_line + (float(i) / steps) * (self.input - base_line) for i in range(1, steps + 1)]
        gradients = self._calc_gradients(scale_inputs, target_label)
        if isinstance(gradients, list):
            ig = []
            for gradient in gradients:
                avg_grad = np.average(gradient[:-1], axis = 0)
                avg_grad = np.transpose(avg_grad, (1, 2, 0))
                dx = (self._preprocess([self.input]) - self._preprocess([base_line])).detach().squeeze(0).cpu().numpy()
                dx = np.transpose(dx, (1, 2, 0))
                ig.append(dx * avg_grad)
        else:
            avg_grad = np.average(gradients[:-1], axis = 0)
            avg_grad = np.transpose(avg_grad, (1, 2, 0))
            dx = (self._preprocess([self.input]) - self._preprocess([base_line])).detach().squeeze(0).cpu().numpy()
            dx = np.transpose(dx, (1, 2, 0))
            ig = dx * avg_grad
        return ig
    
    def _calc_gradients(self, scale_inputs, target_label):
        def calc_grads(input, model, out):
            model.zero_grad()
            calc_grad = torch.sum(out)
            calc_grad.backward(retain_graph=True)
            grads = input.grad.detach().cpu().numpy()
            input.grad = None
            return grads
        
        x = self._preprocess(scale_inputs)
        
        if "yolo" in self.model_name:
            # x_copy = copy(x)
            grads = []
            result = []
            out, logit = self.model(x)
            for pred in out:
                score_li = []
                for anchor_index, cls in target_label:
                    score_li.append(pred[anchor_index][int(cls)+5])
                grads.append(torch.stack(score_li))
            grad_list = torch.stack(grads,dim=1)
            for grad in grad_list:
                result.append(calc_grads(x, self.model, grad))
            return result
            
        else:
            out = self.model(x)
            out = F.softmax(out, dim=1)
            if target_label is None:
                target_label = torch.argmax(out, 1).item()
            index = np.ones((out.size()[0], 1)) * target_label
            index = torch.tensor(index, dtype=torch.int64, device=self.device)
            out = out.gather(1, index)
            return calc_grads(x, self.model, out)
        