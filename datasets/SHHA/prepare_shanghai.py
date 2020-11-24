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

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

path = osp.join('../../../data/ShanghaiTech/part_A', '')

split = ['train', 'test']
for s in split:
    output_path = osp.join('../../../ProcessedData', 'Shanghai_A', s)
    train_path_img = osp.join(output_path, 'img')
    train_path_den = osp.join(output_path, 'den')
    train_path_label = osp.join(output_path, 'label')
    train_path_den_img = osp.join(output_path, 'den_img')

    image_path = osp.join(path, s + '_data', 'images')

    imgs = [f for f in os.listdir(image_path) if osp.isfile(osp.join(image_path, f))]

    gt_path = osp.join(path, s + '_data', 'ground-truth')

    mkdirs(output_path)
    mkdirs(train_path_img)
    mkdirs(train_path_den)
    mkdirs(train_path_label)
    mkdirs(train_path_den_img)

    num_images = len(imgs)

    for idx in range(num_images):
        img_name = imgs[idx]
        img = cv2.imread(osp.join(image_path, img_name))
        height, width, _ = img.shape

        gt = loadmat(osp.join(gt_path, 'GT_' + img_name.replace('jpg', 'mat')))
        positions = gt["image_info"][0, 0][0, 0][0]

        hm = np.zeros(img.shape[:2], dtype=np.float32)

        num_people = len(positions)

        print(idx + 1, num_people)
        for pos in positions:
            draw_gaussian(hm, pos, 75, 15.0, mode='max')

        csv_name = img_name.replace('IMG_', '').replace('jpg', 'csv')
        np.savetxt(osp.join(train_path_den, csv_name), hm, delimiter=",")

        hm = hm * 255
        hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(osp.join(train_path_den_img, img_name.replace('IMG_', '')), hm)

        cv2.imwrite(osp.join(train_path_img, img_name.replace('IMG_', '')), img)


        f = open(osp.join(train_path_label, img_name.replace('IMG_', '').replace('jpg', 'txt')), 'w+')
        for pos in positions:
            label_str = '{:.6f} {:.6f}\n'.format(pos[0] / width, pos[1] / height)
            f.write(label_str)
        f.close()
