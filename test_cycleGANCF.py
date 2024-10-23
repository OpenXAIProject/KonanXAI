import torch
import numpy as np
import cv2

# weight_path = "./resnet50_military_data_epoch10_cuda.pt"

# model = torch.hub.load('pytorch/vision:v0.11.0', 'resnet50')
# weight = torch.load(weight_path)
# key = list(weight.keys())[-1]
# print(key)
# print(weight[key].shape[0])

import torchvision
import torch.nn as nn
from KonanXAI.models.model_import import model_import 
from KonanXAI.datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image 
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.cluster import KMeans


framework = 'torch'
source = 'torchvision'
repo_or_dir = None
data_path = "../dataset/MNIST/raw"
data_type = 'MNIST'
model_name = 'resnet50'
data_resize = [224,224]
cache_or_local = None
weight_path = None   # weight_path = None 이면 pretrained=True 자동으로 들어가게 일단 해놓을까?
cfg_path = None

device = torch.device('cuda:0')
# model = model_import(framework, source, repo_or_dir,
#                                   model_name, cache_or_local, 
#                                   weight_path)
# dataset = load_dataset(framework, data_path = data_path,
#                                     data_type = data_type, resize = data_resize)

model = torchvision.models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
in_channel = model.fc.in_features
model.fc = nn.Linear(in_channel, 10)
model = model.to(device)

train_dataset = torchvision.datasets.MNIST(root = '../dataset/MNIST',
                                     train=True, transform = transforms.ToTensor(), download=False)

test_dataset = torchvision.datasets.MNIST(root = '../dataset/MNIST',
                                     train=False, download=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle = True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)

total = 0
correct = 0

# for epoch in range(10):
#     for i, data in enumerate(train_loader,1):
#         images, labels = data
#         images = images.cuda()
#         labels = labels.cuda()

#         optimizer.zero_grad()
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         loss = criterion(outputs, labels)
#         if (i %100 == 0):
#             print(f'Epoch: {epoch} Batch: {i} loss: {loss.item()}')
        
#         loss.backward()
#         optimizer.step()

path = './resnet50_mnist_epoch10.pt'
# torch.save(model.state_dict(), path)

model.load_state_dict(torch.load(path))
model.to(device)

origin_data = train_dataset[0][0]
origin_label = train_dataset[0][1]

model.eval()
cf_image = origin_data
origin_data = origin_data.cuda().unsqueeze(0)

class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, g):
        gate = self.conv_gate(g)
        x = self.conv_x(x)
        attention = self.softmax(gate)
        x_att = x * attention
        return x_att

class Generator(nn.Module):
    def __init__(self, gf, channels):
        super(Generator, self).__init__()
        self.channels = channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, gf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(gf, gf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(gf * 2, gf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(gf * 4, gf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 8)
        )

        self.attn_layer = nn.ModuleList([
            AttentionGate(gf * 2**(i), gf * 2**(i+1))
            for i in range(3)
        ])

        # Upsampling layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(gf * 8, gf * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 4)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(gf * 8, gf * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(gf * 4, gf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(gf * 2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Downsampling
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        
        # Upsampling
        u1 = self.deconv1(d4)
        u1 = self.attn_layer[2](d3, u1)
        
        u2 = self.deconv2(u1)
        u2 = self.attn_layer[1](d2, u2)
        
        u3 = self.deconv3(u2)
        u3 = self.attn_layer[0](d1, u3)
        
        output = self.deconv4(u3)
        
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return optimizer


class Discriminator(pl.LightningModule):
    def __init__(self, df):
        super(Discriminator, self).__init__()
        self.df = df
        # Define the layers for the discriminator
        self.conv_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if i == 0 else df * 2**(i-1), df * 2**i, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, df * 2**i)) for i in range(4)])
        
        self.final_conv = nn.Conv2d(df * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        out = x
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        validity = self.final_conv(out)
        return validity

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return optimizer
