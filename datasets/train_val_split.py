import os
import os.path as osp
import sys
import random
import shutil

def mkdir(d):
    if not osp.exists(d):
        os.makedirs(d)

seed = 123
random.seed(seed)
data_root = sys.argv[1]

train_img_path = osp.join(data_root, 'train', 'img')
train_den_path = osp.join(data_root, 'train', 'den_img')
train_label_path = osp.join(data_root, 'train', 'label')

val_img_path = osp.join(data_root, 'val', 'img')
val_den_path = osp.join(data_root, 'val', 'den_img')
val_label_path = osp.join(data_root, 'val', 'label')

mkdir(val_img_path)
mkdir(val_den_path)
mkdir(val_label_path)

data_files = [filename for filename in os.listdir(train_img_path) \
                if os.path.isfile(os.path.join(train_img_path, filename))]
num_samples = len(data_files) 

random.shuffle(data_files)

val_len = int(num_samples * 0.1)

val_files = data_files[:val_len]

for file in val_files:
	label_file = osp.splitext(file)[0] + '.txt'
	img_file = file
	den_file = file
	shutil.move(osp.join(train_img_path, img_file), osp.join(val_img_path, img_file))
	shutil.move(osp.join(train_den_path, den_file), osp.join(val_den_path, den_file))
	shutil.move(osp.join(train_label_path, label_file), osp.join(val_label_path, label_file))
