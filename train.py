import sys
sys.path.append('datasets')
import torch
import os
import os.path as osp
from datetime import datetime
from trainer import Trainer
from config import config

config.set_scope('Train')

batch_size = config.getint('batch_size')

dataset = config.get('dataset')

if dataset == 'SHHA':
    from datasets.SHHA.loading_data import loading_data
if dataset == 'SHHB':
    from datasets.SHHB.loading_data import loading_data
if dataset == 'MALL':
    from datasets.Mall.loading_data import loading_data
if dataset == 'JHU':
    from datasets.JHU.loading_data import loading_data
if dataset == 'QNRF':
    from datasets.QNRF.loading_data import loading_data

train_loader, val_loader, _ = loading_data(batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_loaders = {
    'val_loader': val_loader,
    'train_loader': train_loader
}

pwd = osp.split(os.path.realpath(__file__))[0]
trainer = Trainer(config, data_loaders, dataset + '_' + datetime.now().strftime("%y%m%d%H%M"), './exp', device)

if config.getbool('load_state'):
    trainer.load_state(config.get('state_path'))

trainer.forward(config.getint('epoch'))
