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
from numpy.lib.stride_tricks import as_strided


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)

dataset_name = 'WorldExpo'
path = osp.join('../../../DBs/CC/WorldExpo')

split = ['train']
for s in split:
    # folders = [f for f in os.listdir(osp.join(path, s)) if not osp.isfile(osp.join(path, s, f))]
    # for sub_test in folders:

    output_path = osp.join('../../../DBs/CC/ProcessedData', 'WorldExpo', s)
    train_path_img = osp.join(output_path, 'img')
    train_path_den = osp.join(output_path, 'den')
    train_path_label = osp.join(output_path, 'label')
    train_path_den_img = osp.join(output_path, 'den_img')

    image_path = osp.join(path, s, 'img')
    gt_path = osp.join(path, s, 'den')

    imgs = [f for f in os.listdir(image_path) if '.jpg' in f and osp.isfile(osp.join(image_path, f))]

    mkdirs(output_path)
    mkdirs(train_path_img)
    mkdirs(train_path_label)
    mkdirs(train_path_den_img)

    num_images = len(imgs)

    for idx in range(num_images):
        img_name = imgs[idx]
        print(img_name)
        img = cv2.imread(osp.join(image_path, img_name))
        height, width, _ = img.shape

        gt = np.loadtxt(osp.join(gt_path, img_name.replace('.jpg', '.csv')), delimiter=',')

        gt_peak = pool2d(gt, 3, 1, 1)
        locations = np.transpose(np.nonzero(gt * (gt_peak == gt)))
        locations = np.insert(locations, 2, values=[int(1e6 * gt[int(locations[i][0]), int(locations[i][1])]) for i in range(len(locations))], axis=1)

        peak = -1
        if len(locations > 0):
            bins = np.bincount(locations[:, 2])
            peak = bins.argmax()

        positions = np.array([[loc[1], loc[0]] for loc in locations if loc[2] >= peak])

        hm = np.zeros(img.shape[:2], dtype=np.float32)

        num_people = len(positions)

        print(idx + 1, num_people)
        for pos in positions:
            draw_gaussian(hm, pos, 51, 3.0, mode='max')

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

