import torchvision.transforms as torch_transforms
from torch.utils.data import DataLoader
import misc.transforms as transforms
from .GCC import GCC
import os.path as osp
import torch
import random



def loading_data(batch_size):
    data_path = '/workspace/DBs/CC/GCC'
    # BGR
    gcc_mean_std = ([0.269087553024, 0.291243076324, 0.302234709263], [0.184846073389, 0.211051672697, 0.227743327618])

    main_transform = transforms.Compose([
        transforms.RandomCrop(height=768, width=1024),
        transforms.RandomAffine(degrees=(-10, 10), translate=(.3, .3), scale=(1., 3.)),
        transforms.RandomHorizontallyFlip()
    ])

    val_main_transform = transforms.Compose([
        transforms.RandomCrop(height=768, width=1024)
    ])

    img_transform = torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(*gcc_mean_std)
    ])


    train_set = GCC(data_path, osp.join(data_path, 'lists', 'training.txt'), 'train', main_transform=main_transform, img_transform=img_transform, downscale=1)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)

    val_set = GCC(data_path, osp.join(data_path, 'lists', 'validation.txt'), 'val', main_transform=val_main_transform, img_transform=img_transform, downscale=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)

    test_set = GCC(data_path, osp.join(data_path, 'lists', 'testing.txt'), 'test', main_transform=val_main_transform, img_transform=img_transform, downscale=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)

    return train_loader, val_loader, test_loader
