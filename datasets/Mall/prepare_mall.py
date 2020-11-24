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

standard_size = (480, 640);

dataset_name = 'MALL'
path = osp.join('../../../data', 'mall_dataset')
output_path = osp.join('../../../ProcessedData', 'Mall')
train_path_img = osp.join(output_path, 'img')
train_path_den = osp.join(output_path, 'den')
train_path_label = osp.join(output_path, 'label')
train_path_den_img = osp.join(output_path, 'den_img')

gt_path = osp.join(path, 'mall_gt.mat')
img_path = osp.join(path, 'frames')

mkdirs(output_path)
mkdirs(train_path_img)
mkdirs(train_path_den)
mkdirs(train_path_label)
mkdirs(train_path_den_img)

data = loadmat(gt_path)
num_images = len(data['frame'][0])

for idx in range(num_images):
    img_name = 'seq_{:06d}.jpg'.format(idx + 1) 
    img = cv2.imread(osp.join(img_path, img_name))

    height, width, _ = img.shape

    hm = np.zeros(standard_size, dtype=np.float32)

    positions = data['frame'][0][idx]['loc'][0][0]
    num_people = len(positions)

    print(idx + 1, num_people)
    for pos in positions:
        draw_gaussian(hm, pos, 15, 4.0)

    csv_name = '{}.csv'.format(idx + 1)
    np.savetxt(osp.join(train_path_den, csv_name), hm, delimiter=",")

    hm = hm * 10000
    cv2.imwrite(osp.join(train_path_den_img, '{}.jpg'.format(idx + 1)), hm)

    cv2.imwrite(osp.join(train_path_img, '{}.jpg'.format(idx + 1)), img)

    f = open(osp.join(train_path_label, '{}.txt'.format(idx + 1)), 'w+')
    for pos in positions:
        label_str = '{:.6f} {:.6f}\n'.format(pos[0] / width, pos[1] / height)
        f.write(label_str)
    f.close()

