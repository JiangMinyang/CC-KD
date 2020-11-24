import torch.nn as nn
import torch
from torchvision import models
from misc.utils import initialize_weights

class CSRNet(nn.Module):
    def __init__(self, frontend_feat, load_weights=False):
        super(CSRNet, self).__init__()
        vgg = models.vgg16(pretrained = True)
        features = list(vgg.features.children())
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self._initialize_weights()

        self.frontend = nn.Sequential(*features[:16], *features[17:23])

    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        # x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  



class CC_HEAD(nn.Module):
    def __init__(self, in_out_channels, dropout_p=0.5, act='sigmoid'):
        super(CC_HEAD, self).__init__()

        # self.base = BasicBlock(in_out_channels, in_out_channels)

        self.fc = nn.Sequential(
                nn.Conv2d(64, 64, 1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1, stride=1, padding=0))

        initialize_weights(self.modules())
        self.activation = nn.Sigmoid()
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
 
    def forward(self, x):
        logit = self.fc(x)
        print(torch.max(logit))

        out = self.activation(logit)

        return out, logit 

def get_feature_net():
    model = CSRNet([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 512, 512, 512])
    return model


def get_CC_head(in_channels=64, act='sigmoid'):
    return CC_HEAD(in_channels, act=act)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.C1 = get_CC_head()
        self.C2 = get_CC_head()

        self.G = get_feature_net()

    def forward(self, img):
        feature = self.G(img)

        output1, logit1 = self.C1(feature)
        output2, logit2 = self.C2(feature)
        return torch.squeeze(output1, 1), torch.squeeze(output2, 1)
