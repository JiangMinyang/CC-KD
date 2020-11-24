import os.path as osp
import cv2
import os
import random
import shutil

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

img_path = '/workspace/DBs/CC/ProcessedData/Demo/test/img'
label_path = '/workspace/DBs/CC/ProcessedData/Demo/test/label'

target_img_path = '/workspace/DBs/CC/ProcessedData/Demo/img'
target_label_path = '/workspace/DBs/CC/ProcessedData/Demo/label'

mkdirs(target_label_path)
mkdirs(target_img_path)

data_files = [filename for filename in os.listdir(img_path) \
                   if os.path.isfile(osp.join(img_path, filename))]

num_samples = len(data_files) 

random.shuffle(data_files)

for i in range(100):
	shutil.copyfile(osp.join(img_path, data_files[i]), osp.join(target_img_path, 'demo%d.jpg' % i))
	shutil.copyfile(osp.join(label_path, data_files[i].replace('jpg', 'txt')), osp.join(target_label_path, 'demo%d.txt' % i))
