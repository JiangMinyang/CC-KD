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

class MOT(data.Dataset):
    def __init__(self, root, list_file, mode, main_transform=None, img_transform=None, downscale=4):

        self.file_folder = []
        self.file_name = []
        self.gt_cnt = []
        self.root = root
        self.downscale = downscale
        
        with open(list_file) as f:
            lines = f.readlines()
        
        for line in lines:
            splited = line.strip().split()

            self.file_folder.append(splited[0])
            self.file_name.append(splited[1])
            self.gt_cnt.append([int(splited[2]), int(splited[3]), int(splited[4])])

        self.mode = mode
        self.main_transform = main_transform  
        self.img_transform = img_transform
        self.num_samples = len(lines)   
        
    
    def __getitem__(self, index):
        img, num_people = self.get_data(index)

        img = img.copy()
        if self.img_transform is not None:
            img = self.img_transform(img)         
        num_people = torch.from_numpy(np.array(num_people))

        if (self.mode == 'test'):
            return {'image': img, 'label': num_people, 'fname': self.file_name[index]}
        else:
            return img, num_people

    def __len__(self):
        return self.num_samples


    def get_data(self, index):
        img = self.read_image(index)
        if self.main_transform is not None:
            img, label = self.main_transform(img, []) 

        height, width, _ = img.shape

        return img, self.gt_cnt[index]

    def __len__(self):
        return self.num_samples

    def read_image(self, index):

        img = cv2.imread(osp.join(self.root, self.file_folder[index], self.file_name[index]) + '.jpg')
        
        return img


    def get_num_samples(self):
        return self.num_samples       
            
        