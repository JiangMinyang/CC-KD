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

import pandas as pd

import math
from scipy import spatial
def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

dataset_name = 'JHU_CROWD'
# path = osp.join('../../../data/ShanghaiTech/part_A', '')
path = osp.join('../../../DBs/CC', 'jhu_crowd_v2.0')

split = ['val', 'test']
for s in split:
    output_path = osp.join('../../../DBs/CC/ProcessedData', 'JHU_CROWD', s)
    train_path_img = osp.join(output_path, 'img')
    train_path_den = osp.join(output_path, 'den')
    train_path_label = osp.join(output_path, 'label')
    train_path_den_img = osp.join(output_path, 'den_img')

    image_path = osp.join(path, s, 'images')

    imgs = [f for f in os.listdir(image_path) if osp.isfile(osp.join(image_path, f))]

    gt_path = osp.join(path, s, 'gt')

    mkdirs(output_path)
    mkdirs(train_path_img)
    mkdirs(train_path_den)
    mkdirs(train_path_label)
    mkdirs(train_path_den_img)

    num_images = len(imgs)

    for idx in range(num_images):
        img_name = imgs[idx]
        if ('jpg' not in img_name):
            continue
        print(idx, osp.join(image_path, img_name))
        img = cv2.imread(osp.join(image_path, img_name))
        height, width, _ = img.shape

        gt_file = img_name.replace('jpg', 'txt') 
        positions = np.loadtxt(osp.join(gt_path, gt_file), dtype=np.float32).reshape(-1, 6)
        num_people = len(positions)
        print(img_name, num_people, positions.shape)
        if len(positions) > 0:
            positions = positions[:, :2]

        hm = np.zeros(img.shape[:2], dtype=np.float32)

        positions = positions.astype(int)

        if (num_people > 0):
            kd_tree = spatial.KDTree(positions)
            for pos in positions:
                distances, _ = kd_tree.query(pos, k=5)
                for i in range(1, 5):
                    if (math.isinf(distances[i])):
                        distances[i] = 75
                r = min(max(3, math.floor(np.mean(distances[1:4]))), 75)
                # if (r % 2 == 0):
                    # r += 1
                draw_gaussian(hm, pos, 101, r / 5, mode='max')
        # print(idx + 1, num_people)
        # for pos in positions:
            # draw_gaussian(hm, pos, 15, 3.0, mode='sum')

        # csv_name = img_name.replace('IMG_', '').replace('jpg', 'csv')
        # np.savetxt(osp.join(train_path_den, csv_name), hm, delimiter=",")

        hm = hm * 255 / np.max(hm)
        hm = hm.astype(np.uint8)
        hm_jet = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        cv2.imwrite(osp.join(train_path_den_img, img_name.replace('.jpg', '_' + str(s) + '.jpg')), hm)
        cv2.imwrite(osp.join(train_path_den_img, img_name.replace('.jpg', '_' + str(s) + '_jet.jpg')), hm_jet)

        cv2.imwrite(osp.join(train_path_img, img_name), img)


        f = open(osp.join(train_path_label, gt_file), 'w+')
        for pos in positions:
            label_str = '{:.6f} {:.6f}\n'.format(pos[0] / width, pos[1] / height)
            f.write(label_str)
        f.close()

