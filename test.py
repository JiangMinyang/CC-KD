import sys
sys.path.append('datasets')

import torch
import os
import os.path as osp
import cv2
import numpy as np
from datetime import datetime
from models.net import Net
from models.decode import Decoder, heatmap_decode, heatmap_decode_hm, heatmap_decode_location

import misc.utils as utils
import torchvision.transforms as torch_transforms
import misc.transforms as transforms
from scipy import signal
from config import config

config.set_scope('Test')

dataset = config.get('dataset')

if dataset == 'SHHA':
    from datasets.SHHA.loading_data import loading_data
    from datasets.SHHA.dataset_config import config as dataset_config
if dataset == 'SHHB':
    from datasets.SHHB.loading_data import loading_data
    from datasets.SHHB.dataset_config import config as dataset_config
if dataset == 'MALL':
    from datasets.Mall.loading_data import loading_data
    from datasets.Mall.dataset_config import config as dataset_config
if dataset == 'JHU':
    from datasets.JHU.loading_data import loading_data
    from datasets.JHU.dataset_config import config as dataset_config
if dataset == 'QNRF':
    from datasets.QNRF.loading_data import loading_data
    from datasets.QNRF.dataset_config import config as dataset_config

#exp_name = 'QNRF_4X_ZOOM_IN_2010090826'
#model_name = 'all_ep_1003_mae_d_110.1.pth'

#exp_name = 'SHHB_4X_ZOOM_IN_2010220705'
#model_name = 'all_ep_173_mae_4.5_mse_7.7_loss_0.5.pth'
#model_name = 'all_ep_7_mae_5.0_mse_8.6_loss_0.4.pth'

#exp_name = 'JHU_4X_ZOOM_IN_2010262142'
#model_name = 'all_ep_215_mae_77.0_mse_275.1_loss_1.0.pth'
#exp_name = 'JHU_4X_ZOOM_IN_2011010705'
#model_name = 'all_ep_149_mae_76.2_mse_270.8_loss_1.0.pth'
#exp_name = 'WE_4X_ZOOM_IN_2011062127'
#model_name = 'latest_state.pth'
# exp_name = 'JHU_4X_ZOOM_IN_2011092210'
# model_name = 'all_ep_13_mae_d_77.6.pth'


#****exp_name = 'SHHB_4X_ZOOM_IN_2010230008'
#****model_name = 'all_ep_123_mae_5.9_mse_9.1_loss_0.5.pth'


# model_path = './exp/SHHA2006300610/all_ep_1153_mae_85.4_mse_148.7_loss_19.1.pth'
# model_path = './exp/SHHA2007020518/all_ep_2_mae_212.6_mse_371.3_loss_45.5.pth'

exp_name = config.get('exp_name')
model_name = config.get('model_name')
model_path = osp.join('./exp', exp_name, model_name)

batch_size = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_, _, test_loader = loading_data(batch_size)

vis = False
save_output = True
output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, exp_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

localization_output_dir = os.path.join(output_dir, 'loc')
if not os.path.exists(localization_output_dir):
    os.mkdir(localization_output_dir)

model_state_dict = torch.load(model_path)

net = Net()
net.load_state_dict(model_state_dict['net'])
net.to(device)
net.eval()

decoder = None
if (config.getbool('use_decoder')):
    decoder = Decoder()
    decoder.load_state_dict(model_state_dict['decoder'])
    decoder.to(device)
    decoder.eval()

mae = 0.0
mse = 0.0

mae_d = 0.0
mse_d = 0.0

count = 0
mode = 1  # 1 for IAM, 2 for regular

def cc_count(model, image, depth, pred_map, decoder=None):
    h, w, _ = image.shape
    if h < w:
        scale_transform = transforms.Scale(height=dataset_config.height, width=dataset_config.width)
    else:
        scale_transform = transforms.Scale(height=dataset_config.width, width=dataset_config.height)

    new_img, _ = scale_transform(image, [])
    to_tensor = torch_transforms.ToTensor()
    img_tensor = to_tensor(new_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred_count, hm = heatmap_decode_hm(output)
        pred_count = pred_count.to('cpu').numpy()[0]
        hm = hm.to('cpu').numpy()[0]
        if (decoder is not None and pred_count > dataset_config.decoder_involve_threshold):
            pred_count = decoder(output).to('cpu').numpy()[0]

        hm_h, hm_w = hm.shape
        if pred_count < dataset_config.test_split_threshold or depth > dataset_config.test_split_depth:
            locations = heatmap_decode_location(image, output)
            for location in locations:
                location[0] = max(min(location[0], h - 1), 0)
                location[1] = max(min(location[1], w - 1), 0)
                pred_map[int(location[0]), int(location[1])] = location[2]
            return pred_count
        else:
            if h > w:
                new_h = h // 2
                img1 = image[:new_h, :]
                img2 = image[new_h:, :]
                pred_map1 = pred_map[:new_h, :]
                pred_map2 = pred_map[new_h:, :]
                hm_count1 = max(1, np.sum(hm[:hm_h // 2, :]))
                hm_count2 = max(1, np.sum(hm[hm_h // 2:, :]))
            else:
                new_w = w // 2
                img1 = image[:, :new_w]
                img2 = image[:, new_w:]
                pred_map1 = pred_map[:, :new_w]
                pred_map2 = pred_map[:, new_w:]
                hm_count1 = max(1, np.sum(hm[:, :hm_w // 2]))
                hm_count2 = max(1, np.sum(hm[:, hm_w // 2:]))

            if (pred_count < 500):
                count1 = cc_count(model, img1, depth + 1, pred_map1, decoder)
                count2 = cc_count(model, img2, depth + 1, pred_map2, decoder)
            else:
                count1 = max(hm_count1, cc_count(model, img1, depth + 1, pred_map1, decoder))
                count2 = max(hm_count2, cc_count(model, img2, depth + 1, pred_map2, decoder))

            return count1 + count2

for batch_idx, data in enumerate(test_loader):
    #count += 1
    img = data['image'].to(device)
    label = data['label']
    fname = data['fname'][0]
    original_img = data['original_img'][0].numpy()
    gt = data['num_people'][0].numpy()

    count += 1
    if mode == 'IAM':
        gt_count = gt
        h, w = original_img.shape[:2]
        pred_map = np.zeros((h, w))
        pred = cc_count(net, original_img, 1, pred_map, None)
        mae += abs(gt_count-pred)
        mse += ((gt_count-pred)*(gt_count-pred))

        print(gt_count, pred)

        csv_name = fname.replace('jpg', 'csv')
        np.savetxt(osp.join(localization_output_dir, csv_name), pred_map, delimiter=',', fmt='%.3f')
        k = np.ones((3, 3))
        bmap = signal.convolve2d((pred_map > 0), k, boundary='symm', mode='same')
        points = np.transpose(np.nonzero(bmap))
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY);
        original_img = cv2.merge([gray,gray,gray])
        for point in points:
            original_img[point[0]][point[1]] = [0, 0, 255]
        cv2.imwrite(osp.join(output_dir, fname.split('.')[0] + '_output_%d' % pred + '.png'), original_img)

    else:
        with torch.no_grad():

            output = net(img)
            pred_count = heatmap_decode(output).to('cpu').numpy()[0]

            if (decoder is not None):
                d_count = decoder(output).to('cpu').numpy()[0]
            else:
                d_count = 0

            density_map = output.cpu().numpy()[0]
            gt_count = gt

            print(gt_count, pred_count, d_count)

            mae += abs(gt_count - pred_count)
            mse += ((gt_count - pred_count) ** 2)

            mae_d += abs(gt_count - d_count)
            mse_d += ((gt_count - d_count) ** 2)

            if save_output:
                utils.save_density_map(density_map, output_dir, fname.split('.')[0] + '_output_' + '_%d' % pred_count + '.png')
                utils.save_density_map(label.numpy()[0], output_dir, fname.split('.')[0] + '_label_' + '_%d' % gt_count + '.png')
        
mae = mae / count
mse = np.sqrt(mse / count)
print('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))


if mode != 1:
    mae_d = mae_d / count
    mse_d = np.sqrt(mse_d / count)
    print('\nMAE_D: %0.2f, MSE_D: %0.2f' % (mae_d, mse_d))

