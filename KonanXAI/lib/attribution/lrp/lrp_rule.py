
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys

def safe_divide(relevance_in, z, eps=sys.float_info.epsilon):
    sign = torch.sign(z)     # 여기 이 부분 부터는 따로 함수로 묶어도 될 듯
    sign[z==0] = 1
    eps = torch.tensor(eps, device='cuda:0')
    z = z + sign*eps
    s = relevance_in / z
    
    return s

import enum

def forward_hook(m, input_tensor, output_tensor):
    setattr(m, 'X', input_tensor[0])
    setattr(m, 'Y', output_tensor[0])


# class LRPRule:
#     self.EPSILON = enum.auto()
#     self.ALPHABETA = enum.auto()

    

class LRPModule:
    def __init__(self, module):
        self.module: nn.Module = module
        self.handle = []
        if isinstance(self.module, nn.Module):
            self.module.register_forward_hook(forward_hook)

    def handle_remove(self):
        for handle in self.handle:
            handle.remove()

    def forward(self, x):
        return self.module.forward(x)
    
    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def epsilon(self, R, rule, alpha):
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
        if rule == 'Epsilon':
            return self.epsilon(R, rule, alpha)
        elif rule == 'AlphaBeta':
            return self.alphabeta(R, rule, alpha)
        
    def __repr__(self):
        try:
            return str(self.module.__class__.__name__)
        except Exception:
            return str(self.__class__.__name__)

class ConvNd(LRPModule):
    def epsilon(self, R, rule, alpha):
        x = self.module.X.clone()
        w = self.module.weight.clone()

        x = x.clamp(min=0)
        w = w.clamp(min=0)
        # if self.power != None:
        #    x = x.pow(self.power)
        #    w = w.pow(self.power)
        conv = {nn.Conv1d: F.conv1d, nn.Conv2d: F.conv2d, nn.Conv3d: F.conv3d}[type(self.module)]
        z = conv(x, weight = w, bias = None, stride = self.module.stride, 
                    padding = self.module.padding, groups = self.module.groups,)
        sign = torch.sign(z)     # 여기 이 부분 부터는 따로 함수로 묶어도 될 듯
        sign[z==0] = 1
        z = z + sign*alpha
        s = R / z
        
        conv_bwd = {nn.Conv1d: F.conv_transpose1d, nn.Conv2d: F.conv_transpose2d, nn.Conv3d: F.conv_transpose3d}[type(self.module)]
        if self.module.stride != (1,1):
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
        return relevance_out
    
    
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
    
    def epsilon(self, R, rule, alpha):
        i = 0
        
        for module in reversed(self.modules):
            # if hasattr(module, 'module'):
            #     print("Calc module :", module.module.__class__.__name__)
            # else:
            #     print("Calc module :", module.__class__.__name__)

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
    
    def alphabeta(self, R, rule, alpha):
        i = 0
        
        for module in reversed(self.modules):
            # if hasattr(module, 'module'):
            #     print("Calc module :", module.module.__class__.__name__)
            # else:
            #     print("Calc module :", module.__class__.__name__)

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

    def epsilon(self, R, rule, alpha):
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
        return torch.add(*x)
    
    def epsilon(self, R, rule, alpha):
        for i, m in enumerate(self.modules):
            if self.X[i] is None:
                if isinstance(m[-1], Input):
                    handle = m[-1].handle.pop().id
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
        
        # 차원 맞추기
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
        
    
class Clone(LRPModule):
    def __init__(self, origin, num=2):
        self.origin = origin
        self.num = num
    # max pool쪽 err 
    def epsilon(self, R, rule, alpha):
        Z = []
        if isinstance(self.origin, nn.modules.pooling._MaxPoolNd):
            X = self.origin(self.origin.X[-1]).unsqueeze(0).detach()
        else:
            X = self.origin.X[-1].detach()
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
    
    

class Cat(LRPModule):
    def __init__(self, module1, module2, dim):
        def add_forward_hook(m, input_tensor, output_tensor):
            self.X.append(output_tensor[0])
        self.X = []
        self.handle = []
        self.handle.append(module1[-1].module.register_forward_hook(add_forward_hook))
        self.handle.append(module2[-1].module.register_forward_hook(add_forward_hook))
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)
    
    def epsilon(self, R, rule, alpha):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        out = []
        for x, c in zip(self.X, C):
            out.append(x * c)
        
        return out

class Flatten(LRPModule):
    def __init__(self, prev_module):
    
        self.X = None
        def flatten_forward_hook(m, input_tensor, output_tensor):
            self.X = output_tensor[0]
        self.handle = []
        self.handle.append(prev_module.register_forward_hook(flatten_forward_hook))

    def epsilon(self, R, rule, alpha):
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

    def epsilon(self, R, rule, alpha):
        return R
    
    def alphabeta(self, R, rule, alpha):
        return R
    
class BatchNormNd(LRPModule):
    def epsilon(self, R, rule, alpha):
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
    pass
    # def epsilon(self, R, rule, alpha):
    #     #unpool 해주기.... 현재 Relevance를 2배로 확장..
    #     maxunpool = {nn.MaxPool1d: F.max_unpool1d, nn.MaxPool2d: F.max_unpool2d, nn.MaxPool3d: F.max_unpool3d}[type(self.module)]
    #     # R = maxunpool(input = R, stride = self.module.stride, padding = self.module.padding, indices=self.module.indices, kernel_size= self.module.kernel_size)
    #     # R = nn.MaxUnpool2d(input = R,stride= self.module.stride)
    #     return R