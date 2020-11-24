import torchvision.transforms as torch_transforms
from torch.utils.data import DataLoader
import misc.transforms as transforms
from datasets.cc_dataloader import CCLoader
import os.path as osp
import torch
from datasets.Mall.dataset_config import config

def loading_data(batch_size):
    data_path = config.data_path

    main_transform = transforms.Compose([
        transforms.Scale(height=config.height, width=config.width),
        transforms.RandomAffine(degrees=config.aug_degrees, translate=config.aug_translate, scale=config.aug_scale),
        transforms.RandomHorizontallyFlip()
    ])

    val_main_transform = transforms.Compose([
        transforms.Scale(height=config.height, width=config.width)
    ])

    img_transform = torch_transforms.Compose([
        torch_transforms.ToTensor(),
    ])

    train_set = CCLoader(osp.join(data_path, 'train'), 'train', main_transform=main_transform, img_transform=img_transform, downscale=4, default_dis=config.padm_default_distance)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=config.dataloader_worker, shuffle=True, drop_last=False)

    val_set = CCLoader(osp.join(data_path, 'val'), 'val', main_transform=val_main_transform, img_transform=img_transform, downscale=4, default_dis=config.padm_default_distance)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=config.dataloader_worker, shuffle=False, drop_last=False)

    test_set = CCLoader(osp.join(data_path, 'test'), 'test', main_transform=val_main_transform, img_transform=img_transform, downscale=4, default_dis=config.padm_default_distance)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=config.dataloader_worker, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader