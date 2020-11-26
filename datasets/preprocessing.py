from PIL import Image
import numpy as np
import os
import os.path as osp
import cv2
import sys

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def cal_new_size_V2(im_h, im_w, min_size, max_size):
    rate = 1.0 * max_size / im_h
    rate_w = im_w * rate
    if rate_w > max_size:
        rate = 1.0 * max_size / im_w
    tmp_h = int(1.0 * im_h * rate / 16) * 16

    if tmp_h < min_size:
        rate = 1.0 * min_size / im_h
    tmp_w = int(1.0 * im_w * rate / 16) * 16

    if tmp_w < min_size:
        rate = 1.0 * min_size / im_w
    tmp_h = min(max(int(1.0 * im_h * rate / 16) * 16, min_size), max_size)
    tmp_w = min(max(int(1.0 * im_w * rate / 16) * 16, min_size), max_size)

    rate_h = 1.0 * tmp_h / im_h
    rate_w = 1.0 * tmp_w / im_w
    assert tmp_h >= min_size and tmp_h <= max_size
    assert tmp_w >= min_size and tmp_w <= max_size
    return tmp_h, tmp_w, rate_h, rate_w

def cal_new_size_V1(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio, 1.0

def generate_image(im_path, min_size, max_size):
    im = Image.open(im_path).convert('RGB')
    im_w, im_h = im.size
    im_h, im_w, rr_h, rr_w = cal_new_size_V1(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr_h != 1.0 or rr_w != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
    return Image.fromarray(im)


if __name__ == '__main__':
    data_root = sys.argv[1]
    if (len(sys.argv) > 2):
        min_size = int(sys.argv[2])
        max_size = int(sys.argv[3])
    else:
        min_size = 512
        max_size = 2048
    img_path = osp.join(data_root, 'img')
    img_out_path = osp.join(data_root, 'img_resized')
    mkdirs(img_out_path)
    data_files = [filename for filename in os.listdir(img_path) \
                           if osp.isfile(osp.join(img_path,filename))]
    for img_file in data_files:
        resized_img = generate_image(osp.join(img_path, img_file), min_size, max_size)
        resized_img.save(osp.join(img_out_path, img_file))