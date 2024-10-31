import torch.nn as nn
import torch.nn.functional as F
import torch
# from models.resnet import BottleNeck, ResNet
from torchvision.models.vgg import VGG
__all__ = ["Attention_VGG", "make_attention_vgg19", "make_attention_vgg16"]
cfg = {
    'vgg16': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],#, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]#, 'M'],
}
class Attention_VGG(VGG):
    def __init__(self, cfg, num_classes):
        features = _make_features(cfg)
        super(Attention_VGG, self).__init__(features=features, num_classes= num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.att_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.after_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_att = nn.BatchNorm2d(512)
        self.att_conv = nn.Conv2d(512, num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3 = nn.Conv2d(num_classes,1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(7)
        self.sigmoid = nn.Sigmoid()
        
                
    def forward(self, x):
        x = self.features(x)
        #add attention branch
        fe = x
        ax = self.bn_att(self.att_layer(x))
        ax = self.relu(self.bn_att2(self.att_conv(ax)))
        self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))
        ax = self.att_conv2(ax)
        ax= self.att_gap(ax)
        ax = ax.view(ax.size(0),-1)
        
        rx = x*self.att
        rx = rx+x
        per = rx
        
        rx = self.after_layer(rx)
        rx = self.avgpool(rx)
        rx = torch.flatten(rx, 1)
        rx = self.classifier(rx)

        return ax, rx, [self.att, fe, per]
    
def _make_features(cfg: "list[str | int]", batchnorm=True):
    layers = []
    in_chn = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_chn, v, kernel_size=3, padding=1))
            if batchnorm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU())
            in_chn = v
    return nn.Sequential(*layers)  

def make_attention_vgg16(num_classes=80):
    assert 'vgg16' in cfg
    return Attention_VGG(cfg['vgg16'], num_classes)

def make_attention_vgg19(num_classes=80):
    assert 'vgg19' in cfg
    return Attention_VGG(cfg['vgg19'], num_classes)
