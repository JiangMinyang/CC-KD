import numpy as np
import os
import os.path as osp
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
import cv2

from draw_gaussian import draw_gaussian
from models.decode import heatmap_decode
import pandas as pd

import math
from scipy import spatial
import misc.utils as utils

class GCC(data.Dataset):
    def __init__(self, root, list_file, mode, main_transform=None, img_transform=None, downscale=4):

        self.crowd_level = []
        self.time = []
        self.weather = []
        self.file_folder = []
        self.file_name = []
        self.gt_cnt = []
        self.root = root
        self.downscale = downscale
        
        with open(list_file) as f:
            lines = f.readlines()
        
        for line in lines:
            splited = line.strip().split()

            self.crowd_level.append(splited[0])
            self.time.append(splited[1])
            self.weather.append(splited[2])
            self.file_folder.append(splited[3])
            self.file_name.append(splited[4])
            self.gt_cnt.append(int(splited[5]))

        self.mode = mode
        self.main_transform = main_transform  
        self.img_transform = img_transform
        self.num_samples = len(lines)   
        
    
    def __getitem__(self, index):
        img, hm, num_people = self.get_data(index)      

        img = img.copy()
        if self.img_transform is not None:
            img = self.img_transform(img)         

        if (self.mode == 'test'):
            return {'image': img, 'label': hm, 'fname': self.file_name[index]}
        else:
            return img, hm

    def __len__(self):
        return self.num_samples


    def get_data(self, index):
        img, label = self.read_image_and_gt(index)
        if self.main_transform is not None:
            img, label = self.main_transform(img, label) 

        height, width, _ = img.shape
        if (len(label) > 0):
            label[:, 0] /= self.downscale
            label[:, 1] /= self.downscale

        out_height = height // self.downscale
        out_width = width // self.downscale
        hm = np.zeros((out_height, out_width), dtype=np.float32)

        positions = label.astype(int)
        num_people = len(positions)

        if (num_people > 0):
            kd_tree = spatial.KDTree(positions)
            for pos in positions:
                distances, _ = kd_tree.query(pos, k=2)
                if (math.isinf(distances[1])):
                    distances[1] = 15
                r = max(3, min(math.floor(distances[1]), 15))
                if (r % 2 == 0):
                    r += 1
                draw_gaussian(hm, pos, r, r / 3, mode='max')
        # if self.mode == 'train':
            # utils.save_density_map(hm, './output', self.file_name[index] + '_gt.png')
            # cv2.imwrite('./output/' + self.file_name[index] + '.png', img)
        hm = torch.from_numpy(hm)

        return img, hm, num_people
    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, index):

        img = cv2.imread(osp.join(self.root, self.file_folder[index][1:], 'pngs', self.file_name[index] + '.png'))

        label = np.loadtxt(osp.join(self.root, self.file_folder[index][1:], 'label', self.file_name[index] + '.txt'), dtype=np.float32).reshape(-1, 2)
        
        return img, label


    def get_num_samples(self):
        return self.num_samples       
            
        