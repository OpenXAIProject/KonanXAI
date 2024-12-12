import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import itertools


from typing import Literal, List, Optional, Callable, Union

import os
import json
import numpy as np
import cv2

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image


from KonanXAI.explainer.counterfactual import Counterfactual
from KonanXAI.model_improvement import Trainer
from KonanXAI.datasets import load_dataset
from KonanXAI.utils import save_tensor

from PIL import Image
import matplotlib.pyplot as plt

__all__ = ["CycleganCF"]


# ABCMeta 상속으로 해야하나?
class CycleganCF(Counterfactual):
    ''' explain something...
    
    '''
    def __init__(self, framework, model, dataset, config):
        Counterfactual.__init__(self, framework, model, dataset, config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cycleGAN_train = config['cycleGAN_train']
        self.CF_cycleGAN_train = config['CF_cycleGAN_train']
        self.batch = config['batch']

        self.gen_AtoB_weight_path = config['gen_AtoB_weight_path']
        self.gen_BtoA_weight_path = config['gen_BtoA_weight_path']
        self.disc_A_weight_path = config['disc_A_weight_path']
        self.disc_B_weight_path = config['disc_B_weight_path']
        self.CF_gen_AtoB_weight_path = config['CF_gen_AtoB_weight_path']
        self.CF_gen_BtoA_weight_path = config['CF_gen_BtoA_weight_path']
        self.CF_disc_A_weight_path = config['CF_disc_A_weight_path']
        self.CF_disc_B_weight_path = config['CF_disc_B_weight_path']
        self.cycleGAN_lr = config['cycleGAN_learning_rate']
        self.cycleGAN_epochs = config['cycleGAN_epochs']
        self.CF_lr = config['CF_learning_rate']
        self.CF_epochs = config['CF_epochs']
        self._lambda = config['lambda'] 
        self.mu = config['mu']
        self.gamma = config['gamma']

        self.data_type = config['data_type']
        self.data_path = config['data_path']
        self.save_path = config['save_path']
        self.data_resize = config['data_resize']

        self.input_dataset = load_dataset(self.framework, data_path = self.data_path,
                                    data_type = self.data_type, resize = self.data_resize, mode = 'explainer', label = self.input_label)
        self.target_dataset = load_dataset(self.framework, data_path = self.data_path,
                                    data_type = self.data_type, resize = self.data_resize, mode = 'explainer', label = self.target_label)
        
    # def _make_cycleGAN_dataset(self):
    #     self.input_dataset = load_dataset(self.framework, data_path = self.data_path,
    #                                 data_type = self.data_type, resize = self.data_resize, mode = 'explainer', label = self.input_label)
    #     self.target_dataset = load_dataset(self.framework, data_path = self.data_path,
    #                                 data_type = self.data_type, resize = self.data_resize, mode = 'explainer')



    def _set_cycleGAN_model(self):
        input_nc = 1
        output_nc = 1
        self.gen_AtoB = Generator(input_nc, output_nc)
        self.gen_BtoA = Generator(output_nc, input_nc)
        self.disc_A = Discriminator(input_nc)
        self.disc_B = Discriminator(output_nc)
        self.gen_AtoB.to(self.device)
        self.gen_BtoA.to(self.device)
        self.disc_A.to(self.device)
        self.disc_B.to(self.device)



    def _load_cycleGAN_weight(self):
        self.gen_AtoB.load_state_dict(torch.load(self.gen_AtoB_weight_path))
        self.gen_BtoA.load_state_dict(torch.load(self.gen_BtoA_weight_path))
        self.disc_A.load_state_dict(torch.load(self.disc_A_weight_path))
        self.disc_B.load_state_dict(torch.load(self.disc_B_weight_path))

        self.gen_AtoB.eval()
        self.gen_BtoA.eval()
        self.disc_A.eval()
        self.disc_B.eval()


    def _load_CFcycleGAN_weight(self):
        self.gen_AtoB.load_state_dict(torch.load(self.CF_gen_AtoB_weight_path))
        self.gen_BtoA.load_state_dict(torch.load(self.CF_gen_BtoA_weight_path))

        self.gen_AtoB.eval()
        self.gen_BtoA.eval()
        
    def _set_cycleGAN_loss_and_optimizer(self):
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()      
        self.criterion_identity = torch.nn.L1Loss()      

        self.optimizer_gen = torch.optim.Adam(itertools.chain(self.gen_AtoB.parameters(), self.gen_BtoA.parameters()),lr = self.cycleGAN_lr, betas = (0.5, 0.999))
        self.optimizer_disc_A = torch.optim.Adam(self.disc_A.parameters(), lr = self.cycleGAN_lr, betas = (0.5, 0.999))
        self.optimizer_disc_B = torch.optim.Adam(self.disc_B.parameters(), lr = self.cycleGAN_lr, betas = (0.5, 0.999))

        self.lr_scheduler_gen = torch.optim.lr_scheduler.LambdaLR(self.optimizer_gen, lr_lambda=LambdaLR(self.cycleGAN_epochs, 0, int(0.8 *self.cycleGAN_epochs)).step)
        self.lr_scheduler_disc_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_disc_A, lr_lambda=LambdaLR(self.cycleGAN_epochs, 0, int(0.8 *self.cycleGAN_epochs)).step)
        self.lr_scheduler_disc_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_disc_B, lr_lambda=LambdaLR(self.cycleGAN_epochs, 0, int(0.8 *self.cycleGAN_epochs)).step)




    def _set_CFcycleGAN_loss_and_optimizer(self):
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()      
        self.criterion_identity = torch.nn.L1Loss()      
        self.criterion_CF = torch.nn.MSELoss()

        self.optimizer_gen = torch.optim.Adam(itertools.chain(self.gen_AtoB.parameters(), self.gen_BtoA.parameters()), lr = self.CF_lr, betas = (0.5, 0.999))
        self.optimizer_disc_A = torch.optim.Adam(self.disc_A.parameters(), lr = self.CF_lr, betas = (0.5, 0.999))
        self.optimizer_disc_B = torch.optim.Adam(self.disc_B.parameters(), lr = self.CF_lr, betas = (0.5, 0.999))

        self.lr_scheduler_gen = torch.optim.lr_scheduler.LambdaLR(self.optimizer_gen, lr_lambda=LambdaLR(self.CF_epochs, 0, int(0.8 *self.CF_epochs)).step)
        self.lr_scheduler_disc_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_disc_A, lr_lambda=LambdaLR(self.CF_epochs, 0, int(0.8 *self.CF_epochs)).step)
        self.lr_scheduler_disc_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_disc_B, lr_lambda=LambdaLR(self.CF_epochs, 0, int(0.8 *self.CF_epochs)).step)


    
    def _forward(self, x, y, c):
        pass

    def _set_target_real_and_fake(self):
        self.target_real = torch.zeros(self.batch).to(self.device)
        self.target_fake = torch.zeros(self.batch).to(self.device)

        for i in range(self.batch):
            self.target_real[i] = self.input_label
            self.target_fake[i] = self.target_label

    def _set_CF_target_real_and_fake(self):
        self.target_real = torch.ones(4).to(self.device)
        self.target_real.requires_grad = False
        self.target_fake = torch.zeros(4).to(self.device)
        self.target_fake.requires_grad = False

    def _train_init(self):
        self.gen_AtoB.train()
        self.gen_BtoA.train()
        self.disc_A.train()
        self.disc_B.train()

        self.gen_AtoB.apply(weights_init_normal)
        self.gen_BtoA.apply(weights_init_normal)
        self.disc_A.apply(weights_init_normal)
        self.disc_B.apply(weights_init_normal)

    def train(self):
        self._train_init()
        self._set_cycleGAN_loss_and_optimizer()
        self._set_target_real_and_fake()

        self.input_dataset.set_train()
        self.target_dataset.set_train()
        self.input_dataset.set_batch(self.batch)
        self.target_dataset.set_batch(self.batch)

        for epoch in range(self.cycleGAN_epochs):
            for i, (input_data, target_data) in enumerate(zip(self.input_dataset, self.target_dataset)):
                if target_data[0].shape[0] == 4:
                    real_A = input_data[0].to(self.device)
                    real_B = target_data[0].to(self.device)

                    self.optimizer_gen.zero_grad()

                    same_B = self.gen_AtoB(real_B)
                    loss_identity_B = self.criterion_identity(same_B, real_B) * self._lambda

                    same_A = self.gen_BtoA(real_A)
                    loss_identity_A = self.criterion_identity(same_A, real_A) * self._lambda

                    fake_B = self.gen_AtoB(real_A)
                    pred_fake = self.disc_B(fake_B)
                    loss_GAN_AtoB = self.criterion_GAN(pred_fake, self.target_real)

                    fake_A = self.gen_BtoA(real_B)
                    pred_fake = self.disc_A(fake_A)
                    loss_GAN_BtoA = self.criterion_GAN(pred_fake, self.target_real)

                    recovered_A = self.gen_BtoA(fake_B)
                    loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A) * self.mu

                    recovered_B = self.gen_AtoB(fake_A)
                    loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B) * self.mu

                    loss_gen = loss_identity_A + loss_identity_B + loss_GAN_AtoB + loss_GAN_BtoA + loss_cycle_ABA + loss_cycle_BAB
                    loss_gen.backward()

                    self.optimizer_gen.step()
                    ######### discriminator_A
                    self.optimizer_disc_A.zero_grad()

                    pred_real = self.disc_A(real_A)
                    loss_disc_real = self.criterion_GAN(pred_real, self.target_real)

                    pred_fake = self.disc_A(fake_A.detach())
                    loss_disc_fake = self.criterion_GAN(pred_fake, self.target_fake)

                    loss_disc_A = (loss_disc_real + loss_disc_fake) * 0.5
                    loss_disc_A.backward()

                    self.optimizer_disc_A.step()

                    ################ discriminator_B
                    self.optimizer_disc_B.zero_grad()

                    pred_real = self.disc_B(real_B)
                    loss_disc_real = self.criterion_GAN(pred_real, self.target_real)

                    pred_fake = self.disc_B(fake_B.detach())
                    loss_disc_fake = self.criterion_GAN(pred_fake, self.target_fake)

                    loss_disc_B = (loss_disc_real + loss_disc_fake) * 0.5
                    loss_disc_B.backward()

                    self.optimizer_disc_B.step()

                    if i % 100 == 0:
                        print('epoch:', epoch, 'i:', i)
                        print({'loss_G': loss_gen, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_AtoB + loss_GAN_BtoA),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_disc_A + loss_disc_B)})

                else:
                    continue

            self.lr_scheduler_gen.step()
            self.lr_scheduler_disc_A.step()
            self.lr_scheduler_disc_B.step()

            self.gen_AtoB_weight_path = './cycleGAN_gen_AtoB.pth'
            self.gen_BtoA_weight_path = './cycleGAN_gen_BtoA.pth'
            self.disc_A_weight_path = './cycleGAN_disc_A.pth'
            self.disc_B_weight_path = './cycleGAN_disc_B.pth'
            torch.save(self.gen_AtoB.state_dict(), self.gen_AtoB_weight_path)
            torch.save(self.gen_BtoA.state_dict(), self.gen_BtoA_weight_path)
            torch.save(self.disc_A.state_dict(), self.disc_A_weight_path)
            torch.save(self.disc_B.state_dict(), self.disc_B_weight_path)
            



    
    def CF_train(self):
        self._train_init()
        self._load_cycleGAN_weight()
        self._set_CFcycleGAN_loss_and_optimizer()
        self._set_CF_target_real_and_fake()

        self.input_dataset.set_train()
        self.target_dataset.set_train()
        self.input_dataset.set_batch(self.batch)
        self.target_dataset.set_batch(self.batch)

        for epoch in range(self.CF_epochs):
            for i, (input_data, target_data) in enumerate(zip(self.input_dataset, self.target_dataset)):
                if target_data[0].shape[0] == 4:
                    real_A = input_data[0].to(self.device)
                    real_B = target_data[0].to(self.device)

                    self.optimizer_gen.zero_grad()

                    same_B = self.gen_AtoB(real_B)
                    loss_identity_B = self.criterion_identity(same_B, real_B) * self._lambda

                    same_A = self.gen_BtoA(real_A)
                    loss_identity_A = self.criterion_identity(same_A, real_A) * self._lambda

                    fake_B = self.gen_AtoB(real_A)
                    pred_fake = self.disc_B(fake_B)
                    loss_GAN_AtoB = self.criterion_GAN(pred_fake, self.target_real)

                    fake_A = self.gen_BtoA(real_B)
                    pred_fake = self.disc_A(fake_A)
                    loss_GAN_BtoA = self.criterion_GAN(pred_fake, self.target_real)

                    recovered_A = self.gen_BtoA(fake_B)
                    loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A) * self.mu

                    recovered_B = self.gen_AtoB(fake_A)
                    loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B) * self.mu

                    class_A_loss = self.model(fake_A)
                    class_B_loss = self.model(fake_B)
                    loss_CF = self.criterion_CF(class_A_loss, torch.ones_like(class_A_loss)) + self.criterion_CF(class_B_loss, torch.zeros_like(class_B_loss))


                    loss_gen = loss_identity_A + loss_identity_B + loss_GAN_AtoB + loss_GAN_BtoA + loss_cycle_ABA + loss_cycle_BAB + self.gamma * loss_CF
                    loss_gen.backward()

                    self.optimizer_gen.step()
                    ######### discriminator_A
                    self.optimizer_disc_A.zero_grad()

                    pred_real = self.disc_A(real_A)
                    loss_disc_real = self.criterion_GAN(pred_real, self.target_real)

                    pred_fake = self.disc_A(fake_A.detach())
                    loss_disc_fake = self.criterion_GAN(pred_fake, self.target_fake)

                    loss_disc_A = (loss_disc_real + loss_disc_fake) * 0.5
                    loss_disc_A.backward()

                    self.optimizer_disc_A.step()

                    ################ discriminator_B
                    self.optimizer_disc_B.zero_grad()

                    pred_real = self.disc_B(real_B)
                    loss_disc_real = self.criterion_GAN(pred_real, self.target_real)

                    pred_fake = self.disc_B(fake_B.detach())
                    loss_disc_fake = self.criterion_GAN(pred_fake, self.target_fake)

                    loss_disc_B = (loss_disc_real + loss_disc_fake) * 0.5
                    loss_disc_B.backward()

                    self.optimizer_disc_B.step()

                    if i % 100 == 0:
                        print('epoch:', epoch, 'i:', i)
                        print({'loss_G': loss_gen, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_AtoB + loss_GAN_BtoA),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_CF': (loss_CF), 'loss_D': (loss_disc_A + loss_disc_B)})

                else:
                    continue

            self.lr_scheduler_gen.step()
            self.lr_scheduler_disc_A.step()
            self.lr_scheduler_disc_B.step()

            self.CF_gen_AtoB_weight_path = './CF_cycleGAN_gen_AtoB.pth'
            self.CF_gen_BtoA_weight_path = './CF_cycleGAN_gen_BtoA.pth'
            self.CF_disc_A_weight_path = './CF_cycleGAN_disc_A.pth'
            self.CF_disc_B_weight_path = './CF_cycleGAN_disc_B.pth'
            torch.save(self.gen_AtoB.state_dict(), self.CF_gen_AtoB_weight_path)
            torch.save(self.gen_BtoA.state_dict(), self.CF_gen_BtoA_weight_path)
            torch.save(self.disc_A.state_dict(), self.CF_disc_A_weight_path)
            torch.save(self.disc_B.state_dict(), self.CF_disc_B_weight_path)
            
    def cycleGAN_inference(self):
        pass

    def apply(self):
        self._set_cycleGAN_model()

        if self.cycleGAN_train == True:
            self.train()

        else:
            if os.path.isfile(self.gen_AtoB_weight_path) == False:
                raise Exception('cycleGAN weight가 없습니다')
            else:
                pass
        
        if self.CF_cycleGAN_train == True:
            self.CF_train()
            
        else:
            if os.path.isfile(self.CF_gen_AtoB_weight_path) == False:
                raise Exception('CF_cycleGAN weight가 없습니다')
            else:
                pass

        self._load_CFcycleGAN_weight()

        for i, (data_A, data_B) in enumerate(zip(self.input_dataset, self.target_dataset)):
            real_A = data_A[0].to(self.device)
            real_B = data_B[0].to(self.device)
            
            fake_B = self.gen_AtoB(real_A).data
            fake_A = 0.5 *(self.gen_BtoA(real_B).data + 1.0)


            if os.path.isdir(self.save_path) == False:
                os.makedirs(self.save_path) 
            list_split_data_A = os.path.split(self.input_dataset.train_items[i][0])
            list_split_data_B = os.path.split(self.target_dataset.train_items[i][0])
            real_A_save_path = self.save_path + '/real_{}'.format(self.input_label) + list_split_data_A[-1]
            fake_B_save_path = self.save_path + '/CF_{}'.format(self.target_label) + list_split_data_A[-1]
            real_B_save_path = self.save_path + '/real_{}'.format(self.target_label) + list_split_data_B[-1]
            fake_A_save_path = self.save_path + '/CF_{}'.format(self.input_label) + list_split_data_B[-1]
            
            save_tensor(real_A, real_A_save_path)
            save_tensor(fake_B, fake_B_save_path)
            save_tensor(real_B, real_B_save_path)
            save_tensor(fake_A, fake_A_save_path)




        

# class cycleganDataset(MNIST):
#     def load_src_path(self, train = True):
#         if train == True:
#             train_path = glob(self.src_path+"/training/*/*.*")
            
#             for path in train_path:
#             label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
#             label = int(label)
#             self.train_items.append((path, label))
#         else:
#             test_path = glob(self.src_path+"/testing/*/*.*")# 수정 필요
#         items = []
        

#         for path in test_path:
#             label = os.path.dirname(path).split(os.sep)[-1].split(".")[0]
#             label = int(label)
#             self.test_items.append((path, label))




class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant(model.bias.data, 0.0)



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

    



