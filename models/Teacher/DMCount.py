import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, config):
        super(VGG, self).__init__()
        out_channels = config['out_channels']
        num_blocks = config['num_blocks']
        self.conv0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), 
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
        self.in_channel = 64
        self.layer1 = self._make_layers(out_channels[1], num_blocks[1])
        self.layer2 = self._make_layers(out_channels[2], num_blocks[2])
        self.layer3 = self._make_layers(out_channels[3], num_blocks[3])
        self.layer4 = self._make_layers(out_channels[4], num_blocks[4])

        self.reg_conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU())
        self.reg_conv2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def _make_layers(self, out_channel, num_block):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(self.in_channel, out_channel, kernel_size=3, padding=1), nn.ReLU()]
        self.in_channel = out_channel
        for i in range(num_block - 1):
            layers += [nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1), nn.ReLU()]

        return nn.Sequential(*layers)


    def forward(self, x, is_feature=False):
        out = self.conv0(x) # 1
        out = self.layer1(out) # 2
        f0 = out
        out = self.layer2(out) # 4
        f1 = out
        out = self.layer3(out) # 8
        f2 = out
        out = self.layer4(out) # 16
        f3 = out
        out = F.upsample_bilinear(out, scale_factor=2) # 8
        out = self.reg_conv1(out)
        out = self.reg_conv2(out)
        reg_f = out
        out = self.density_layer(out)
        B, C, H, W = out.size()
        out_sum = out.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        out_normed = out / (out_sum + 1e-6)
        if (is_feature):
            return [f0, f1, f2, f3, reg_f], out, out_normed
        return out, out_normed

# def _make_layers(out_channels, num_blocks, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
# cfg = {
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
# }
config = {
    'out_channels': (64, 128, 256, 512, 512),
    'num_blocks': (2, 2, 4, 4, 4)
}
def vgg19(model_path=None):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    # model = VGG(make_layers(cfg['E']))
    model = VGG(config)
    model_layers = list(model.state_dict().keys())[:32]
    if model_path is None:
        pre_trained = model_zoo.load_url(model_urls['vgg19'])
    else:
        pre_trained = torch.load(model_path)
    pre_trained_layers = list(pre_trained.keys())[:32]

    for i in range(32):
        model.state_dict()[model_layers[i]].copy_(pre_trained[pre_trained_layers[i]].data)
    # model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
