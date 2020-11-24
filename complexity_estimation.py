from ptflops import get_model_complexity_info
import torch
from models.net import Net
from models.decode import Decoder
import torchvision.models as models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
macs, params = get_model_complexity_info(net, (3, 960, 1280), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
