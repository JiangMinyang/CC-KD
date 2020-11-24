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

# standard_size = (480, 640);

dataset_name = 'QNRF'
path = osp.join('../../../DBs/CC/UCF-QNRF_ECCV18')

split = ['train', 'test']
for s in split:
    output_path = osp.join('../../../DBs/CC/ProcessedData', 'QNRF', s)
    train_path_img = osp.join(output_path, 'img')
    train_path_den = osp.join(output_path, 'den')
    train_path_label = osp.join(output_path, 'label')
    train_path_den_img = osp.join(output_path, 'den_img')

    image_path = osp.join(path, s)

    imgs = [f for f in os.listdir(image_path) if '.jpg' in f and osp.isfile(osp.join(image_path, f))]

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

        gt = loadmat(osp.join(image_path, img_name.replace('.jpg', '_ann.mat')))
        positions = gt["annPoints"]

        hm = np.zeros(img.shape[:2], dtype=np.float32)

        num_people = len(positions)

        print(idx + 1, num_people)
        for pos in positions:
            draw_gaussian(hm, pos, 50, 15.0, mode='max')

#        csv_name = img_name.replace('img_', '').replace('jpg', 'csv')
#        np.savetxt(osp.join(train_path_den, csv_name), hm, delimiter=',', fmt='%.3f')

        hm = hm * 255
        hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(osp.join(train_path_den_img, img_name.replace('img_', '')), hm)

        cv2.imwrite(osp.join(train_path_img, img_name.replace('img_', '')), img)


        f = open(osp.join(train_path_label, img_name.replace('img_', '').replace('jpg', 'txt')), 'w+')
        for pos in positions:
            label_str = '{:.6f} {:.6f}\n'.format(pos[0] / width, pos[1] / height)
            f.write(label_str)
        f.close()

