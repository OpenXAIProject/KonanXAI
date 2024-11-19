import torch.nn as nn
import torch.nn.functional as F
import torch
# from models.resnet import BottleNeck, ResNet
from torchvision.models.resnet import ResNet, Bottleneck
__all__ = ["Attention_ResNet","make_attention_resnet50"]
class Attention_ResNet(ResNet):
    def __init__(self, block, num_classes, layers):
        super(Attention_ResNet, self).__init__( block = block, layers = layers, num_classes = num_classes)
        self.inplanes = 1024
        self.relu = nn.ReLU(inplace=True)
        self.att_layer4 = self._make_layer(block, 512, layers[3], stride = 1, dilate=False)
        self.bn_att = nn.BatchNorm2d(512 * block.expansion)
        self.att_conv = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3 = nn.Conv2d(num_classes,1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(14)
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.conv2_x(x)
        # x = self.conv3_x(x)
        # x = self.conv4_x(x)
        #abn Network 추가
        fe = x
        ax = self.bn_att(self.att_layer4(x))
        ax = self.relu(self.bn_att2(self.att_conv(ax)))
        # bs, cs, ys, xs = ax.shape
        self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))
        # self.att = self.att.view(bs, 1, ys, xs)
        ax = self.att_conv2(ax)
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)

        rx = x * self.att
        rx = rx + x
        per = rx
     
        rx = self.layer4(rx)
        rx = self.avgpool(rx)
        rx = torch.flatten(rx,1)
        rx = self.fc(rx)
        return ax,rx,[self.att, fe, per]
    
def make_attention_resnet50(num_classes=80):
    return Attention_ResNet(Bottleneck, num_classes, [3, 4, 6, 3])
