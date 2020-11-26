from torch.utils.data import DataLoader
from datasets.SHHA.SHHA import SHHA
from torch.utils.data.dataloader import default_collate
from misc.utils import train_collate
import os.path as osp
import torch

def loading_data(args):
    train_set = SHHA(osp.join(args.data_dir, 'train'), args.crop_size, downsample_ratio=8, method='train')
    val_set = SHHA(osp.join(args.data_dir, 'val'), args.crop_size, downsample_ratio=8, method='val')
    test_set = SHHA(osp.join(args.data_dir, 'test'), args.crop_size, downsample_ratio=8, method='test')

    train_loader = DataLoader(train_set, collate_fn=train_collate, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, collate_fn=default_collate, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    test_loader = DataLoader(val, collate_fn=default_collate, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}
