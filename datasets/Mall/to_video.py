import cv2
import numpy as np
import os
import os.path as osp

img_array = []
# img_path = '/media/minyang/Data_SD/DBs/CC/ProcessedData/Mall/whole/img'
#img_path = '/home/minyang/Documents/workspace/CC_DA/output/density_maps_all_ep_87_mae_1'
#img_path = '/home/minyang/Documents/workspace/CC_DA/output/density_maps_all_ep_87_mae_1'
pred_path = '/workspace/CC_DA/output/density_maps_all_ep_87_mae_1/loc'
img_path = '/workspace/CC_DA/output/density_maps_all_ep_87_mae_1'
gt_path = '/workspace/DBs/CC/ProcessedData/Mall/whole/label'

data_files = [filename for filename in os.listdir(img_path) \
                           if 'jpg' in filename and os.path.isfile(os.path.join(img_path,filename))]

data_files = sorted(data_files, key=lambda fname: int(fname.split('.')[0]))
for filename in data_files:
    print(filename)
    gt = len(np.loadtxt(osp.join(gt_path, filename.replace('jpg', 'txt')), delimiter=' ', dtype=np.float32))
    preds = np.loadtxt(osp.join(pred_path, filename.replace('jpg', 'csv')), delimiter=',', dtype=np.float32)
    preds_locs = np.transpose(np.nonzero(preds)).astype(float)

    pred_count = len(preds_locs)
    print(gt, pred_count)
    img = cv2.imread(osp.join(img_path, filename))
    img = cv2.putText(img, 'GT: %d' % gt, (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    img = cv2.putText(img, 'Count: %d' % pred_count, (440, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'XVID'), 2, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()