import json
import sys
sys.path.append('..')
import numpy as np
import os
import os.path as osp
import csv
from scipy.io import loadmat
import cv2
from draw_gaussian import draw_gaussian
import random
from scipy import spatial
import math

def mkdir(d):
    if not osp.exists(d):
        os.makedirs(d)

standard_size = (1080, 1920)
random.seed(123)

dataset_name = 'GCC'
data_root = '../../../DBs'
path = osp.join(data_root, 'GCC')

data_list_file = osp.join(path, 'lists', 'all_list.txt')

with open(data_list_file) as f:
    lines = f.readlines()

partial_list = open(osp.join(path, 'lists', 'partial_list.txt'), 'w+')

for line in lines:
    splited = line.strip().split()

    file_folder = splited[3][1:]
    file_name = splited[4]
    gt_cnt = int(splited[5])

    img_path = osp.join(path, file_folder, 'pngs', file_name + '.png')
    label_path = osp.join(path, file_folder, 'jsons', file_name + '.json')

    print(img_path)
    if not osp.exists(img_path):
        continue

    path_den = osp.join(path, file_folder, 'den')
    path_label = osp.join(path, file_folder, 'label')
    path_den_img = osp.join(path, file_folder, 'den_img')

    mkdir(path_den)
    mkdir(path_label)
    mkdir(path_den_img)

    img = cv2.imread(img_path)
    height, width, _ = img.shape

    with open(label_path, 'r') as f:
        positions = np.array(json.load(f)['image_info'])

    hm = np.zeros(img.shape[:2], dtype=np.float32)

    num_people = len(positions)

    print(file_folder, file_name, num_people, gt_cnt)

    if (num_people > 0):
        positions = positions[:, ::-1]
        for pos in positions:
            hm = draw_gaussian(hm, pos, 15, 4, mode='sum')


    csv_name = file_name + 'csv'
    np.savetxt(osp.join(path_den, csv_name), hm, delimiter=",")

    print(np.max(hm))
    hm = hm / np.max(hm) * 255
    hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(osp.join(path_den_img, file_name + '.jpg'), hm)

    f = open(osp.join(path_label, file_name + '.txt'), 'w+')
    for pos in positions:
        label_str = '{:.6f} {:.6f}\n'.format(pos[0] / width, pos[1] / height)
        f.write(label_str)
    f.close()
partial_list.close()