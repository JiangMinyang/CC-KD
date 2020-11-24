import torchvision.transforms as torch_transforms
from torch.utils.data import DataLoader
import misc.transforms as transforms
from .MOT import MOT
import os.path as osp
import torch
import random



def loading_data(batch_size, target=False):
    data_path = '/workspace/DBs/MOT_DB/MOT17'
    # BGR
    mot17_mean_std = ([0.42776264, 0.45414798, 0.4687067], [0.23127819, 0.23341223, 0.2332089])

    main_transform = transforms.Compose([
            transforms.Scale(height=768, width=1024),
            transforms.RandomAffine(degrees=(-10, 10), translate=(.3, .3), scale=(1., 3.)),
            transforms.RandomHorizontallyFlip()
    ])


    val_main_transform = transforms.Compose([
            transforms.Scale(height=768, width=1024)
    ])

    if target:
        main_transform = val_main_transform

    img_transform = torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(*mot17_mean_std)
    ])



    train_set = MOT(data_path, osp.join(data_path, 'train.txt'), 'train', main_transform=main_transform, img_transform=img_transform, downscale=1)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=False)

    val_set = MOT(data_path, osp.join(data_path, 'test.txt'), 'val', main_transform=val_main_transform, img_transform=img_transform, downscale=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader
