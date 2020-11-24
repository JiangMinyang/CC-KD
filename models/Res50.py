import torch.nn as nn
import torch
from torchvision import models

import torch.nn.functional as F
from misc.utils import *

# model_path = '../PyTorch_Pretrained/resnet50-19c8e357.pth'

class Res50(nn.Module):
    def __init__(self,  pretrained=True):
        super(Res50, self).__init__()

        res = models.resnet50(pretrained=pretrained)
        # pre_wts = torch.load(model_path)
        # res.load_state_dict(pre_wts)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1
        )
        self.own_reslayer_2 = make_res_layer(Bottleneck, 128, 4, stride=1, inplanes=256)        
        self.own_reslayer_2.load_state_dict(res.layer2.state_dict())

        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)        
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self,x):

        
        x = self.frontend(x)

        x = self.own_reslayer_2(x)

        x = self.own_reslayer_3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)   


def make_res_layer(block, planes, blocks, stride=1, inplanes=512):

    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)  


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out        


class CC_HEAD(nn.Module):
    def __init__(self):
        super(CC_HEAD, self).__init__()


        self.fc = nn.Sequential(nn.Conv2d(1024, 128, 1, 1, padding=0),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 1, 1, 1, padding=0))

        self.sigmoid = nn.Sigmoid()

        initialize_weights(self.modules())

        # self.upsample = nn.Upsample(mode='bilinear', scale_factor=4, align_corners=False)
 
    def forward(self, x):
        logit = self.fc(x)

        out = self.sigmoid(logit)

        # out = self.upsample(out)
        return out, logit


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.C1 = CC_HEAD()
        self.C2 = CC_HEAD()
        initialize_weights(self.modules())
        
        self.G = Res50()

    def forward(self, img):
        feature = self.G(img)

        output1, logit1 = self.C1(feature)
        output2, logit2 = self.C2(feature)
        return torch.squeeze(output1, 1), torch.squeeze(output2, 1)
