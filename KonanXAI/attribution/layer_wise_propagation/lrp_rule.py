
import torch
from torchvision.ops import StochasticDepth as stocastic
import torch.nn.functional as F
import torch.nn as nn
import sys
import numpy as np
def safe_divide(relevance_in, z, eps=1e-9):
    sign = torch.sign(z)
    sign[z==0] = 1
    eps = torch.tensor(eps, device='cuda:0')
    z = z + sign*eps
    s = relevance_in / z
    
    return s

import enum
def normalize_0_1(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if min_val == max_val:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)

def z_score_Norm(tensor):
    mean_value = torch.mean(tensor)
    std_value = torch.std(tensor)
    return (tensor-mean_value)/ std_value

def forward_hook(m, input_tensor, output_tensor):
    setattr(m, 'X', input_tensor[0])
    setattr(m, 'Y', output_tensor[0])

class LRPModule:
    def __init__(self, module):
        self.module: nn.Module = module
        self.handle = []
        self.act_idx = -1
        if isinstance(self.module, nn.Module):
            self.module.register_forward_hook(forward_hook)
    def clear(self):
        self.act_idx = -1
    def handle_remove(self):
        for handle in self.handle:
            handle.remove()

    def forward(self, x):
        return self.module.forward(x)
    
    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def epslion(self, R, rule, alpha):
        if len(self.module.X.shape) == 3:
            self.module.X = self.module.X.unsqueeze(0)
        Z = self.forward(self.module.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.module.X, S)[0]

        if torch.is_tensor(self.module.X) == False:
            outputs = []
            outputs.append(self.module.X[0] * C)
            outputs.append(self.module.X[1] * C)
        else:
            outputs = self.module.X * (C)
        return outputs
    
    def signed_forward(self, x, weight):
        return self.forward(x)
        

    def alphabeta(self, R, rule, alpha):
        beta = 1 - alpha
    
        X = self.module.X 
        Z = self.forward(X)

        S = safe_divide(R, Z)
        

        relevance_out = X * self.gradprop(Z, X, S)[0]
        
        return relevance_out
    
    # Abstract
    def backprop(self, R, rule, alpha):
        if rule.lower() == 'epslion':
            return self.epslion(R, rule, alpha)
        elif rule.lower() == 'AlphaBeta':
            return self.alphabeta(R, rule, alpha)
        
    def __repr__(self):
        try:
            return str(self.module.__class__.__name__)
        except Exception:
            return str(self.__class__.__name__)

class ConvNd(LRPModule):
    def epslion(self, R, rule, alpha):
        x = self.module.X.clone()
        w = self.module.weight.clone()

        x = x.clamp(min=0)
        w = w.clamp(min=0)
        conv = {nn.Conv1d: F.conv1d, nn.Conv2d: F.conv2d, nn.Conv3d: F.conv3d}[type(self.module)]
        z = conv(x, weight = w, bias = None, stride = self.module.stride, 
                    padding = self.module.padding, groups = self.module.groups,)
        sign = torch.sign(z)
        sign[z==0] = 1
        z = z + sign*alpha
        if isinstance(R, dict):
            key, rel = next(iter(R.items()))
            rel = rel.detach()
            s = rel / z
        else:
            s = R / z
        
        conv_bwd = {nn.Conv1d: F.conv_transpose1d, nn.Conv2d: F.conv_transpose2d, nn.Conv3d: F.conv_transpose3d}[type(self.module)]
        if self.module.stride != (1,1):
            if isinstance(R, dict):
                _, _, H, W = R[str(key)].size()
            else:
                _, _, H, W = R.size()
            Hnew = (H - 1) * self.module.stride[0] - 2*self.module.padding[0] +\
                        self.module.dilation[0]*(self.module.kernel_size[0]-1) +\
                        self.module.output_padding[0]+1
            Wnew = (W - 1) * self.module.stride[1] - 2*self.module.padding[1] +\
                        self.module.dilation[1]*(self.module.kernel_size[1]-1) +\
                        self.module.output_padding[1]+1
            _, _, Hin, Win = x.size()
            
            cp = conv_bwd(s, weight=w, bias=None, padding=self.module.padding, 
                            output_padding=(Hin-Hnew, Win-Wnew), stride=self.module.stride,
                            dilation=self.module.dilation, groups=self.module.groups,)
            
        else:
            cp = conv_bwd(s, weight=w, bias=None, padding=self.module.padding, 
                            stride=self.module.stride, groups=self.module.groups,)
            
        relevance_out = cp * x
        if isinstance(R, dict):
            R[str(key)] = relevance_out
        return R if isinstance(R, dict) else relevance_out
    
    
    def signed_forward(self, x, weight):
        conv = {nn.Conv1d: F.conv1d, nn.Conv2d: F.conv2d, nn.Conv3d: F.conv3d}[type(self.module)]
        z = conv(x, weight = weight, bias = None, stride = self.module.stride, 
                    padding = self.module.padding, groups = self.module.groups,)
        return z
    
    def alphabeta(self, R, rule, alpha):
        beta = 1 - alpha
        x_pos = self.module.X.clamp(min=0)
        x_neg = self.module.X.clamp(max=0)
        w_pos = self.module.weight.clamp(min=0)
        w_neg = self.module.weight.clamp(max=0)

        def forward(x_pos, x_neg, w_pos, w_neg):
            z1 = self.signed_forward(x_pos, w_pos)
            z2 = self.signed_forward(x_neg, w_neg)

            s1 = safe_divide(R, z1)
            s2 = safe_divide(R, z2)

            c1 = x_pos * self.gradprop(z1, x_pos, s1)[0]
            c2 = x_neg * self.gradprop(z2, x_neg, s2)[0]
            return c1 + c2

        activator = forward(x_pos, x_neg, w_pos, w_neg)
        inhibitor = forward(x_pos, x_neg, w_neg, w_pos)

        relevance_out = alpha * activator + beta * inhibitor

        return relevance_out
    
class Linear(LRPModule):
    def signed_forward(self, x, weight):
        z = F.linear(x, weight)
        return z
    
    def alphabeta(self, R, rule, alpha):
        beta = 1 - alpha
        x_pos = self.module.X.clamp(min=0)
        x_neg = self.module.X.clamp(max=0)
        w_pos = self.module.weight.clamp(min=0)
        w_neg = self.module.weight.clamp(max=0)

        def forward(x_pos, x_neg, w_pos, w_neg):
            z1 = self.signed_forward(x_pos, w_pos)
            z2 = self.signed_forward(x_neg, w_neg)

            s1 = safe_divide(R, z1)
            s2 = safe_divide(R, z2)

            c1 = x_pos * self.gradprop(z1, x_pos, s1)[0]
            c2 = x_neg * self.gradprop(z2, x_neg, s2)[0]
            return c1 + c2

        activator = forward(x_pos, x_neg, w_pos, w_neg)
        inhibitor = forward(x_pos, x_neg, w_neg, w_pos)

        relevance_out = alpha * activator + beta * inhibitor

        return relevance_out

class Sequential(LRPModule):
    def __init__(self, modules):
        self.modules = []
        self._add_modules(modules)
    
    def clear(self):
        for v in self.modules:
            v.clear()

    def _add_modules(self, modules):
        for module in modules:
            if isinstance(module, (str)):
                continue
            if not hasattr(module, 'layer_count'):
                setattr(module, 'layer_count', 0)
            if isinstance(module, LRPModule):
                self.modules.append(module)
            elif isinstance(module, nn.modules.conv._ConvNd):
                self.modules.append(ConvNd(module))
            elif isinstance(module, nn.modules.linear.Linear):
                self.modules.append(Linear(module))
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                self.modules.append(BatchNormNd(module))
            elif isinstance(module, nn.ReLU):
                self.modules.append(ReLU(module, module.layer_count))
            elif isinstance(module, nn.SiLU):
                self.modules.append(SiLU(module, module.layer_count))
            elif isinstance(module, nn.Sigmoid):
                self.modules.append(Sigmoid(module, module.layer_count))
            elif isinstance(module, list):
                self._add_modules(module)
            elif isinstance(module, nn.modules.dropout._DropoutNd):
                self.modules.append(Dropout(module))
            elif isinstance(module, nn.modules.pooling._AdaptiveAvgPoolNd):
                self.modules.append(AdaptiveAvgPoolNd(module))
            elif isinstance(module, nn.modules.pooling._MaxPoolNd):
                self.modules.append(MaxpoolNd(module))
            else:
                self.modules.append(LRPModule(module))
            module.layer_count += 1


    def __len__(self):
        return len(self.modules)
    
    def __getitem__(self, idx):
        return self.modules[idx]
    
    def __repr__(self):
        repr = "Sequential (\n"
        for i, module in enumerate(self.modules):
            repr += f"  ({i}) " + str(module) + "\n"
        repr += ")\n"
        return repr

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def epslion(self, R, rule, alpha):
        for index, module in enumerate(reversed(self.modules)):

            R = module.backprop(R, rule, alpha)
        
        return R if isinstance(R, dict) else R.detach()
    
    def alphabeta(self, R, rule, alpha):
        i = 0
        
        for module in reversed(self.modules):
            if isinstance(R, (list, tuple)):
                s = None
                for r in R:
                    if s is None:
                        s = torch.sum(r)
                    else:
                        s = torch.add(s, torch.sum(r))
                # print("Input Relevance :", s)
            else:
                pass
                # print("Input Relevance :", torch.sum(R))

            R = module.backprop(R, rule, alpha)

            if isinstance(R, (list, tuple)):
                s = None
                for r in R:
                    if s is None:
                        s = torch.sum(r)
                    else:
                        s = torch.add(s, torch.sum(r))
                # print("Output Relevance :", s)
            else:
                pass
                # print("Output Relevance :", torch.sum(R))
        
        return R.detach()

    
class ReLU(LRPModule):
    def __init__(self, module, idx):
        self.module: nn.Module = module
        self.module.X = []
        self.module.Y = []
        self.handle = []
        self.idx = idx
        if not hasattr(self.module, 'hook_handle'):
            def relu_forward_hook(m, input_tensor, output_tensor):
                m.X.append(input_tensor[0])
                m.Y.append(output_tensor[0])
            self.module.hook_handle = self.module.register_forward_hook(relu_forward_hook)

    def epslion(self, R, rule, alpha):
        X = self.module.X[self.idx]
        Z = self.forward(X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, X, S)[0]

        outputs = X * (C)

        return outputs

    def alphabeta(self, R, rule, alpha):
        return R
class SiLU(LRPModule):
    def __init__(self, module, idx):
        self.module: nn.Module = module
        self.module.X = []
        self.module.Y = []
        self.handle = []
        self.idx = idx
        if not hasattr(self.module, 'hook_handle'):
            def silu_forward_hook(m, input_tensor, output_tensor):
                m.X.append(input_tensor[0])
                m.Y.append(output_tensor[0])
            self.module.hook_handle = self.module.register_forward_hook(silu_forward_hook)

    def epslion(self, R, rule, alpha):
        X = self.module.X[self.idx]
        if isinstance(R,dict):
            key, rel = next(iter(R.items()))
        Z = self.forward(X)
        if isinstance(R, dict):
            S = safe_divide(rel, Z)
        else:
            S = safe_divide(R, Z)
        C = self.gradprop(Z, X, S)[0]

        outputs = X * (C)
        
        if isinstance(R, dict):
            R[str(key)] = outputs
        return R if isinstance(R,dict) else outputs
    
    def alphabeta(self, R, rule, alpha):
        return R

class Sigmoid(LRPModule):
    def __init__(self, module, idx):
        self.module: nn.Module = module
        self.module.X = []
        self.module.Y = []
        self.handle = []
        self.idx = idx
        if not hasattr(self.module, 'hook_handle'):
            def sigmoid_forward_hook(m, input_tensor, output_tensor):
                m.X.append(input_tensor[0])
                m.Y.append(output_tensor[0])
            self.module.hook_handle = self.module.register_forward_hook(sigmoid_forward_hook)

    def epslion(self, R, rule, alpha):
        X = self.module.X[self.idx]
        Z = self.forward(X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, X, S)[0]

        outputs = X * (C)

        return outputs
    
    def alphabeta(self, R, rule, alpha):
        return R
class Add(LRPModule):
    def __init__(self, module1: Sequential, module2: Sequential):
        # super().__init__(self)
        def add_forward_hook1(m, input_tensor, output_tensor):
            self.X[0] = F.relu(output_tensor[0].detach())
        def add_forward_hook2(m, input_tensor, output_tensor):
            self.X[1] = F.relu(output_tensor[0].detach())
        self.X = [None, None]
        self.handle = []
        self.modules = [module1, module2]
        self.tensor = None
        self.act_idx = -1
        for i, m in enumerate(self.modules):
            if hasattr(m[-1], 'module') and isinstance(m[-1].module, nn.Module):
                method = add_forward_hook1 if i == 0 else add_forward_hook2
                self.handle.append(m[-1].module.register_forward_hook(method))
            if isinstance(m[-1], Sequential):
                method = add_forward_hook1 if i == 0 else add_forward_hook2
                self.handle.append(m[-1][-1].module.register_forward_hook(method))
            if isinstance(m[-1], StochasticDepth):
                method = add_forward_hook1 if i == 0 else add_forward_hook2
                self.handle.append(m[-1].prev_module.register_forward_hook(method))
        
    def forward(self, x):
        return torch.add(*x)
    
    def epslion(self, R, rule, alpha):
        for i, m in enumerate(self.modules):
            if i == 0 and m[-1].module._get_name().lower() == 'silu':
                layer_index = m[-1].idx
                self.X[0] = m[-1].module.Y[layer_index]
            if self.X[i] is None:
                if isinstance(m[-1], Input):
                    handle_x1 = None
                    for index, handle_idx in enumerate(reversed(self.modules[-1].modules[0].handle)):
                        if self.X[0].shape == m[-1].X[handle_idx.id].squeeze(0).shape or self.X[0].shape == m[-1].X[handle_idx.id].shape:
                            jump_idx = self.modules[-1].modules[0].act_idx
                            handle_x1 = self.modules[-1].modules[0].handle[jump_idx].id
                            self.modules[-1].modules[0].act_idx -= 1
                            break
                    if handle_x1 == None:
                        handle_x1 = self.modules[-1].modules[0].handle[self.modules[-1].modules[0].act_idx].id
                        self.modules[-1].modules[0].act_idx -= 1
                    self.X[i] = m[-1].X[handle_x1].detach()
                    break
            elif self.X[-1] is not None:
                if len(self.modules[-1].modules[0].handle) == 0:
                    break
                else:
                    handle_x1 = self.modules[-1].modules[0].handle[self.modules[-1].modules[0].act_idx].id
                    self.modules[-1].modules[0].act_idx -= 1
                    break
        for x in self.X:
            x.requires_grad_(True)
        
        R = R.squeeze()
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)[0]
        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C)
            outputs.append(self.X[1] * C)
        else:
            outputs = self.X * (C)
        R = outputs

        mR = ()
        for i, m in enumerate(self.modules):
            mR = mR + (m.backprop(R[i], rule, alpha), )

        return mR
    
    def signed_forward(self, x):
        return self.forward(*x)
    
    def alphabeta(self, R, rule, alpha):
        for i, m in enumerate(self.modules):
            if self.X[i] is None:
                if isinstance(m[-1], Input):
                    self.X[i] = m[-1].X.detach()
                    break
        for x in self.X:
            x.requires_grad_(True)

        R = R.squeeze()

        X = self.X
        Z = self.forward(self.X)
        Z_pos = Z.clamp(min=0)
        Z_neg = Z.clamp(max=0)
            
        S_1 = safe_divide(R, Z_pos)
        S_2 = safe_divide(R, Z_neg)

        C_11 = X[0] * self.gradprop(Z_pos, X[0], S_1)[0]
        C_21 = X[1] * self.gradprop(Z_pos, X[1], S_1)[0]
        C_12 = X[0] * self.gradprop(Z_neg, X[0], S_2)[0]
        C_22 = X[1] * self.gradprop(Z_neg, X[1], S_2)[0]

        mR = ()

        for i, m in enumerate(self.modules):
            mR = mR + (m.backprop(R[i], rule, alpha), )

        return mR
    
    def __repr__(self):
        return 'Add : ' + str(self.modules)

class Mul(LRPModule):
    def __init__(self, module1: Sequential, module2: Sequential):
        def add_forward_hook1(m, input_tensor, output_tensor):
            self.X[0] = F.relu(output_tensor[0].detach())
        def add_forward_hook2(m, input_tensor, output_tensor):
            self.X[1] = F.relu(output_tensor[0].detach())
        self.X = [None, None]
        self.handle = []
        self.modules = [module1, module2]
        for i, m in enumerate(self.modules):
            if hasattr(m[-1], 'module') and isinstance(m[-1].module, nn.Module):
                method = add_forward_hook1 if i == 0 else add_forward_hook2
                self.handle.append(m[-1].module.register_forward_hook(method))
            if isinstance(m[-1], Sequential):
                method = add_forward_hook1 if i == 0 else add_forward_hook2
                self.handle.append(m[-1][-1].module.register_forward_hook(method))
            
    def forward(self, x):
        return torch.mul(*x)
    
    def epslion(self, R, rule, alpha):
        for i, m in enumerate(self.modules):
            if self.X[i] is None:
                if isinstance(m[-1], Input):
                    for index, handle_idx in enumerate(reversed(self.modules[-1].modules[0].handle)):
                        if self.X[0].shape[0] == m[-1].X[handle_idx.id].squeeze(0).shape[0] or self.X[0].shape[1] == m[-1].X[handle_idx.id].shape[1]:
                            index +=1
                            handle = self.modules[-1].modules[0].handle.pop(-index).id
                            break
                    self.X[i] = m[-1].X[handle].detach()
                    break
            elif self.X[-1] is not None:
                if len(self.modules[-1].modules[0].handle) == 0:
                    break
                else:
                    self.modules[-1].modules[0].handle.pop()
                    break

        for x in self.X:
            x.requires_grad_(True)
        
        R = R.squeeze()
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)#[0]
        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C)
        R = outputs

        mR = ()
        for i, m in enumerate(self.modules):
            mR = mR + (m.backprop(R[i], rule, alpha), )

        return mR
    
    def signed_forward(self, x):
        return self.forward(*x)
    
    def alphabeta(self, R, rule, alpha):
        for i, m in enumerate(self.modules):
            if self.X[i] is None:
                if isinstance(m[-1], Input):
                    self.X[i] = m[-1].X.detach()
                    break
        for x in self.X:
            x.requires_grad_(True)

        R = R.squeeze()

        X = self.X
        Z = self.forward(self.X)
        Z_pos = Z.clamp(min=0)
        Z_neg = Z.clamp(max=0)
            
        S_1 = safe_divide(R, Z_pos)
        S_2 = safe_divide(R, Z_neg)

        C_11 = X[0] * self.gradprop(Z_pos, X[0], S_1)[0]
        C_21 = X[1] * self.gradprop(Z_pos, X[1], S_1)[0]
        C_12 = X[0] * self.gradprop(Z_neg, X[0], S_2)[0]
        C_22 = X[1] * self.gradprop(Z_neg, X[1], S_2)[0]

        mR = ()

        for i, m in enumerate(self.modules):
            mR = mR + (m.backprop(R[i], rule, alpha), )

        return mR
    
    def __repr__(self):
        return 'Mul : ' + str(self.modules)
    
class Clone(LRPModule):
    def __init__(self, origin, idx=None, num=2):
        self.origin = origin
        self.idx = idx
        self.num = num
    def epslion(self, R, rule, alpha):
        Z = []
        if isinstance(self.origin, tuple):
            self.origin = (self.origin[0].X.detach() + self.origin[1].X.detach())
        if isinstance(self.origin, nn.modules.pooling._MaxPoolNd):
            X = self.origin.Y.unsqueeze(0).detach()
        elif len(R)>2:
            X = self.origin.Y.detach()
        else:
            if self.idx != None:
                X = self.origin.X[self.idx].detach()
            else:
                X = self.origin.X[-1].detach()
        if len(X.shape) == 3:
            X = X.unsqueeze(0)
        X.requires_grad = True
        for _ in range(self.num):
            Z.append(X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, X, S)[0]
        
        R = X * C
        return R
    
    def alphabeta(self, R, rule, alpha):
        Z = []
        X = self.origin.X.detach()
        X.requires_grad = True
        for _ in range(self.num):
            Z.append(X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, X, S)[0]
        R = X * C
        
        return R
    
class Detect(LRPModule):
    def __init__(self, sel_layer, module):
        self.sel_layers = sel_layer
        self.module = []
        if isinstance(module,(list,tuple)):
            for m in module:
                if isinstance(m, nn.modules.conv._ConvNd):
                    self.module.append(ConvNd(m))
    def epslion(self, R, rule, alpha):
        res = {}
        for index, (key, relevance) in enumerate(R.items()):
            relevance = torch.cat([relevance[..., i] for i in range(relevance.size(-1))],dim=1)
            rel = self.module[index].epslion(relevance, rule, alpha)
            res[key] = rel
        return dict(sorted(res.items(),reverse=True))
    
class Cat(LRPModule):
    def __init__(self, *modules, dim = 1):
        self.X = []
        self.handle = []
        self.dim = dim
        self.modules = modules
        super().__init__(self)
        
    def forward(self, x, dim):
        return torch.cat(x, dim)
    
    def epslion(self, R, rule, alpha):
        for v in self.module.modules:
            if hasattr(v.modules[-1],"modules"):
                if hasattr(v.modules[-1].modules[-1], "idx"):
                    index = v.modules[-1].modules[-1].idx
                    self.X.append(v.modules[-1].modules[-1].module.Y[index].unsqueeze(0))
                else:
                    input_handle = v[-1][-1].modules[-1].modules[0].handle[v[-1][-1].modules[-1].modules[0].act_idx].id
                    input_tensor = v[-1][-1].modules[-1].modules[0].X[input_handle]
                    seq_tensor = v.modules[-1].modules[-1].modules[0].forward(input_tensor)
                    add_tensor = seq_tensor+input_tensor
                    self.X.append(add_tensor)
            elif hasattr(v.modules[-1],"module"):
                if hasattr(v.modules[-1],"idx"):
                    index = v.modules[-1].idx
                    self.X.append(v.modules[-1].module.Y[index].unsqueeze(0))
                elif isinstance(v.modules[-1],MaxpoolNd):
                    self.X.append(v.modules[-1].module.forward(self.X[-1]))
                else:
                    self.X.append(v.modules[-1].module.Y.unsqueeze(0))
            
        Z = self.forward(self.X, self.dim)
        if isinstance(R, dict):
            key, rel = next(iter(R.items()))
            S = safe_divide(rel, Z)
        else:    
            S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        out = []
        for x, c in zip(self.X, C):
            out.append(x * c)
        mR = ()
        for i, m in enumerate(self.modules):
            mR = mR + (m.backprop(out[i], rule, alpha), )
        self.X.clear()
        return mR
    
class Route(LRPModule):
    def __init__(self, cat0, cat1):
        self.cat0 = cat0
        self.cat1 = cat1
    def epslion(self, R, rule, alpha):
        split = R.shape[1] // 2
        rel = {}
        rel[str(self.cat0)] = R[:,:split,:,:]
        rel[str(self.cat1)] = R[:,split:,:,:]
        return rel
  
class Upsample(LRPModule):
      def __init__(self, param:dict):
          self.param = param
          self.module = nn.Upsample(**param)
          pass
      def epslion(self, R, rule, alpha):
        invert_upsample = {
            1: F.avg_pool1d,
            2: F.avg_pool2d,
            3: F.avg_pool3d}[2]
        if isinstance(self.param['scale_factor'], float):
            ks = int(self.param['scale_factor'])
        if isinstance(R, dict):
            key, rel = next(iter(R.items()))
            inverted = invert_upsample(R[str(key)], kernel_size = ks, stride = ks)
        else:
            inverted = invert_upsample(R, kernel_size = ks, stride = ks)
        inverted *= ks**2
        relevance = inverted
        R['upsample'] = relevance
        del R[key]
        return R
class StochasticDepth(LRPModule):
    def __init__(self, prev_module, p, mod, training):
        self.X = None
        self.p = p
        self.module = stocastic(p,mod)
        self.prev_module = prev_module
        def stochastic_forward_hook(m, input_tensor, output_tensor):
            self.X = output_tensor[0]
            self.Y = input_tensor[0][0]
        self.handle = []
        self.handle.append(prev_module.register_forward_hook(stochastic_forward_hook))

    def epslion(self, R, rule, alpha):
        R = R.reshape(self.X.shape)
        scaled_out = self.X * self.p
        denominator = scaled_out + 1e-9 * torch.sign(scaled_out)
        relevance_ratio = R / denominator
        R = self.Y * relevance_ratio
        R = R.reshape(self.X.shape)
        return R
    
    def alphabeta(self, R, rule, alpha):
        R = R.reshape(self.X.shape)
        return R
class Flatten(LRPModule):
    def __init__(self, prev_module):
    
        self.X = None
        def flatten_forward_hook(m, input_tensor, output_tensor):
            self.X = output_tensor[0]
        self.handle = []
        self.handle.append(prev_module.register_forward_hook(flatten_forward_hook))

    def epslion(self, R, rule, alpha):
        R = R.reshape(self.X.shape)
        return R
    
    def alphabeta(self, R, rule, alpha):
        R = R.reshape(self.X.shape)
        return R
    
class Input(LRPModule):
    def __init__(self):
        
        self.X = {}
        self.handle = []
        super().__init__(self)

    def epslion(self, R, rule, alpha):
        return R
    
    def alphabeta(self, R, rule, alpha):
        return R

class BatchNormNd(LRPModule):
    def epslion(self, R, rule, alpha):
        X = self.module.X
        weight = self.module.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.module.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2)).pow(0.5))
        Z = X * weight + 1e-9
        S = safe_divide(R, Z)
        Ca = S * weight
        R = self.module.X * (Ca)
        return R

    def alphabeta(self, R, rule, alpha):
        X = self.module.X
        weight = self.module.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
                (self.module.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2)).pow(0.5))

        Z = X * weight + 1e-9
        S = safe_divide(R, Z)
        Ca = S * weight
        R = self.module.X * (Ca)

        return R
    
class Dropout(LRPModule):
    def alphabeta(self, R, rule, alpha):
        
        return R

class AdaptiveAvgPoolNd(LRPModule):
    pass
    # def alphabeta(self, R, rule, alpha):
    #     beta = 1 - alpha

    #     print(R.sum())
    #     X = self.module.X
    #     Z = self.forward(X)

    #     Z_pos = Z.clamp(min=0)
    #     Z_neg = Z.clamp(max=0)
    
    #     S_pos = safe_divide(R, Z_pos)
    #     S_neg = safe_divide(R, Z_neg)

    #     C_pos = X * self.gradprop(Z_pos, X, S_pos)[0]
    #     C_neg = X * self.gradprop(Z_neg, X, S_neg)[0]
        
    

    #     activator = self.forward(C_pos)
    #     inhibitor = self.forward(C_neg)

    #     relevance_out = alpha * activator + beta * inhibitor

    #     return relevance_out
    
class MaxpoolNd(LRPModule):
    def epslion(self, R, rule, alpha):
        X = self.module.X.detach().clone()
        if len(X.shape) == 3:
            X = X.unsqueeze(0)
        X.requires_grad = True
        Z = self.forward(X)  
        S = safe_divide(R, Z, alpha)
        C = self.gradprop(Z, X, S)[0]
        relevance = X * C
        return relevance
