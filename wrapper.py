import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import initialize_weights

def conv1x1_bn(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )
class wrapper(nn.Module):

    def __init__(self, module, in_out_channels, mask=None):
        super(wrapper, self).__init__()
        self.projs = []
        self.mask = np.ones(len(in_out_channels)) if mask is None else mask
        for i in range(len(self.mask)):
            # if (self.mask[i] == 1):
            seq = conv1x1_bn(in_out_channels[i][0], in_out_channels[i][1])
            setattr(self, 'proj{}'.format(i), seq)
        initialize_weights(self.modules())
        self.backbone = module
        self.mse = nn.MSELoss()
        self.cd_loss = CDLoss()

    def forward(self, x, is_feature=True):
        features, out, out_normed = self.backbone(x, is_feature=True)
        for i in range(len(self.mask)):
            if self.mask[i] == 1:
                features[i] = getattr(self, 'proj{}'.format(i))(features[i])

        return features, out, out_normed
        
    def feature_loss(self, features1, features2, loss_type='mse'):
        loss = 0
        # print('=' * 10)
        for i in range(len(self.mask)):
            if self.mask[i] == 1:
                if (loss_type == 'cd'):
                    f_loss = self.cd_loss(features1[i], features2[i])
                else:
                    f_loss = self.mse(features1[i], features2[i])
                # print(i, f_loss)
                loss += f_loss
        # print('=' * 10)
        return loss

class CDLoss(nn.Module):
    """Channel Distillation Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, s, t):
        s = s.mean(dim=(2, 3), keepdim=False)
        t = t.mean(dim=(2, 3), keepdim=False)
        loss = torch.mean(torch.pow(s - t, 2))
        return loss