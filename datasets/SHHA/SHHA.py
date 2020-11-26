from datasets.crowd import Base
import os
import os.path as osp
from PIL import Image
import numpy as np

class SHHA(Base):
    def __init__(self, root_path, crop_size, downsample_ratio=8, method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.img_path = root_path + '/img_resized'
        self.label_path = root_path + '/label'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
        print('number of img: {}'.format(self.num_samples))
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        img_file = self.data_files[item]
        gt_file = img_file.replace('jpg', 'txt')
        img = Image.open(osp.join(self.img_path, img_file)).convert('RGB')
        keypoints = np.loadtxt(osp.join(self.label_path, gt_file), dtype=np.float32).reshape(-1, 2)
        w, h = img.size
        keypoints[:, 0] *= w
        keypoints[:, 1] *= h

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val' or self.method == 'test':
            keypoints = np.loadtxt(osp.join(self.label_path, gt_file), dtype=np.float32).reshape(-1, 2)
            img = self.trans(img)
            name = img_file.split('.')[0]
            return img, len(keypoints), name
