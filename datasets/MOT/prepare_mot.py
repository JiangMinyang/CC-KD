import sys
import os
import os.path as osp

dataset_root = sys.argv[1]


train_writer = open(osp.join(dataset_root, 'train_list.txt'), 'w+')
test_writer = open(osp.join(dataset_root, 'test_list.txt'), 'w+')

writers = {'train': train_writer, 'test': test_writer}


for t in ['train', 'test']:
    video_root = osp.join('images', t)
    videos = os.listdir(osp.join(dataset_root, video_root))
    for video in videos:
        if (('17' in video or '20' in video) and 'FRCNN' not in video):
            continue
        imgs = os.listdir(osp.join(dataset_root, video_root, video, 'img1'))
        for img in imgs:
            p = osp.join(video_root, video, 'img1')
            gt_path = osp.join('labels_with_ids', t, video, 'img1', img.replace('jpg', 'txt'))

            with open(osp.join(dataset_root, gt_path)) as f:
                lines = f.readlines()

            count_0 = len([line for line in lines if int(line.split(' ')[0]) == 1 and float(line.split(' ')[6]) > 0])
            count_20 = len([line for line in lines if int(line.split(' ')[0]) == 1 and float(line.split(' ')[6]) > 0.2])
            count_40 = len([line for line in lines if int(line.split(' ')[0]) == 1 and float(line.split(' ')[6]) > 0.4])
            writers[t].write('{} {} {} {} {}\n'.format(p, img.split('.')[0], count_0, count_20, count_40))

train_writer.close()
test_writer.close()