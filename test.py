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
from torch.utils.data import DataLoader
import torch.optim as optim


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

path = './resnet50_mnist_epoch10'
# torch.save(model.state_dict(), path)

model.load_state_dict(torch.load(path))


origin_data = train_dataset[0][0]
target_data = train_dataset[1][0]
origin_label = train_dataset[0][1]
target_label = train_dataset[1][1]
print(train_dataset[0][1], train_dataset[1][1])

model.eval()

origin_data = origin_data.cuda().unsqueeze(0)
target_data = target_data.cuda().unsqueeze(0)


origin_data.requires_grad = True
target_data.requires_grad = True
print(origin_data.shape)
pred = model(origin_data)
gradients = torch.autograd.grad(pred, origin_data, grad_outputs = torch.ones_like(pred).to(device))[0]
print(gradients.shape)


learning_rate = 0.001
cf_image = origin_data
pdistance = nn.PairwiseDistance(p=0.2, eps = 1e-06, keepdim = False)
optimizer_cf = torch.optim.Adam([cf_image], lr = 0.001)
best_loss = 1e-2
best_cf = torch.rand(1,1,28,28)
best_cf_label = 5
l = 0.1

for iteration in range(500):
    pred = model(cf_image)
    gradients = torch.autograd.grad(pred, cf_image, grad_outputs=torch.ones_like(pred).to(device))[0]
    _, predicted = torch.max(pred.data, 1)
    target_label = [target_label]
    target_label = torch.tensor(target_label, device = 'cuda:0')
    cf_image.requires_

    loss = l * criterion(pred, target_label) + pdistance(cf_image, target_data).sum()
    
    if loss < best_loss:
        best_loss = loss
        best_cf = cf_image
        best_cf_label = predicted

    
    loss.backward()
    optimizer_cf.step()
    #cf_image = cf_image - gradients
    cf_label = predicted
    if (iteration %100 == 0):
        print(f'loss {loss} best_cf_label {best_cf_label} predicted {predicted}')



best_cf = best_cf.squeeze(0).detach().cpu().numpy()
best_cf = np.array(best_cf*255).transpose(1,2,0)
best_cf = cv2.cvtColor(best_cf, cv2.COLOR_BGR2RGB)
path = './resnet50_mnist_best_cf.jpg'
cv2.imwrite(best_cf, path)
