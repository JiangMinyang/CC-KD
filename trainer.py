import os
import os.path as osp
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from models.net import Net
from models.losses import FocalLoss
from misc.utils import *
from models.decode import heatmap_decode, Decoder
import cv2
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, config, dataloaders, exp_name, exp_path, device, mode='train'):
        self.config = config

        self.exp_name = exp_name
        self.exp_path = exp_path

        self.print_interval = config.getint('print_interval')
        self.validation_interval = config.getint('validation_interval')
        self.lr = config.getfloat('lr')

        self.net = Net()
        self.decoder = Decoder()

        self.device = device
        self.net.to(device)
        self.decoder.to(device)

        self.optimizer_d = optim.Adam(self.decoder.parameters(), lr=self.lr, weight_decay=5e-4)
        self.optimizer_g = optim.Adam(self.net.G.parameters(), lr=self.lr, weight_decay=5e-4)
        self.optimizer_c = optim.Adam(self.net.C1.parameters(), lr=self.lr, weight_decay=5e-4)

        self.scheduler_g = StepLR(self.optimizer_g, step_size=100, gamma=0.95)
        self.scheduler_c = StepLR(self.optimizer_c, step_size=100, gamma=0.95)
        self.scheduler_d = StepLR(self.optimizer_d, step_size=100, gamma=0.95)

        self.criterion1 = FocalLoss()
        self.MSE = nn.MSELoss()

        self.train_record = {'best_mae': 1e20, 'best_mse':1e20, 'best_model_name': '', 'best_mae_d':1e20}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

        self.epoch = 0
        self.i_tb = 0
        self.num_batch = 0

        self.val_loader = dataloaders['val_loader']

        self.train_loader = dataloaders['train_loader']
        
        if mode == 'train' and not osp.exists(self.exp_path):
            os.mkdir(self.exp_path)

        self.writer = SummaryWriter(osp.join(self.exp_path, self.exp_name))

    def load_state(self, path):
        latest_state = torch.load(path)
        self.net.load_state_dict(latest_state['net'])
        self.decoder.load_state_dict(latest_state['decoder'])

    def adjust_learning_rate(self, optimizer, epoch):
        self.lr = self.original_lr
        for i in range(len(self.steps)):
            
            scale = self.scales[i] if i < len(self.scales) else 1
            
            
            if epoch >= self.steps[i]:
                self.lr = self.lr * scale
                if epoch == self.steps[i]:
                    break
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

    def reset_grad(self):
        self.optimizer_g.zero_grad()
        self.optimizer_c.zero_grad()
        self.optimizer_d.zero_grad()

    def scheduler_step(self):
        self.scheduler_g.step()
        self.scheduler_c.step()
        self.scheduler_d.step()

    def forward(self, epoches):
        for epoch in range(epoches):
            self.epoch = epoch

            # training    
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('='*20 )

            # validation
            if epoch % self.validation_interval == 0:
                self.timer['val time'].tic()
                self.test('source')
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

            self.scheduler_step() 


    def train(self): # training for all datasets
        self.net.train()

        for batch_idx, data in enumerate(self.train_loader):
            self.timer['iter time'].tic()
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)

            self.reset_grad()

            # step A
            output = self.net(img)
            loss = self.criterion1(output, label)
            loss.backward()

            self.optimizer_g.step()
            self.optimizer_c.step()
            self.reset_grad()

            gt_count = heatmap_decode(label).detach()
            pred_count = self.decoder(output.detach())
            decoder_loss = self.MSE(pred_count, gt_count)
            decoder_loss.backward()
            self.optimizer_d.step()
            self.reset_grad()

            if (batch_idx + 1) % self.print_interval == 0:
                self.i_tb += 1

                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)

                # gt_count = gt_count.to('cpu').numpy()[0]
                gt_count = heatmap_decode(label).to('cpu').numpy()[0]
                loc_count = heatmap_decode(output).to('cpu').numpy()[0]
                pred_count = pred_count.detach().to('cpu').numpy()[0]

                print('Train Epoch: {} [{}/{}]\tLoss1: {:.6f}\t decoder_loss: {:.6f}\t time {:.2f}'.format(
                self.epoch, batch_idx, self.num_batch, loss.item(), decoder_loss.item(), self.timer['iter time'].diff))
                print('[Source cnt: gt: {:.1f} pred: {:.2f} decode_net: {:.6f}'.format(gt_count, loc_count, pred_count))
        if self.num_batch == 0:
            self.num_batch = batch_idx

    def test(self, dataloader=None):
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        maes_d = AverageMeter()
        mses_d = AverageMeter()

        dataloader = self.val_loader

        fig = plt.figure(figsize=(16, 100))
        counter = 0

        for batch_idx, data in enumerate(dataloader):
            counter += self.batch_size
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                output = self.net(img)

                loss = self.criterion(output, label)

                pred_count = self.decoder(output.detach())
                gt_count = heatmap_decode(label)
                decoder_loss = self.MSE(pred_count, gt_count)

                gt_count = gt_count.to('cpu').numpy()
                pred_cnt = heatmap_decode(output).to('cpu').numpy()
                decoder_cnt = pred_count.detach().to('cpu').numpy()

                losses.update(loss.item())
                d_loss.update(decoder_loss.item())

                for i in range(output1.shape[0]):
                    print(gt_count[i], pred_cnt[i], decoder_cnt[i])
                    maes.update(abs(gt_count[i] - pred_cnt[i]))
                    maes_d.update(abs(gt_count[i] - decoder_cnt[i]))

                    mses.update((gt_count[i] - pred_cnt[i]) ** 2)
                    mses_d.update((gt_count[i]  - decoder_cnt[i]) ** 2)

                density_maps = output.detach().to('cpu').numpy()
                label_maps_c = label.to('cpu').numpy()
                if (counter > 40):
                    continue
                for i in range(len(density_maps)):
                    label_map_c = 255 * label_maps_c[i] / np.max(label_maps_c[i])
                    density_map = 255 * density_maps[i] / np.max(density_maps[i])
                    ax = fig.add_subplot(100, 2, 2 * (batch_idx * self.batch_size + i) + 1)
                    ax.imshow(density_map, cmap='gray')
                    ax.axis('off')
                    ax2 = fig.add_subplot(100, 2, 2 * (batch_idx * self.batch_size + i) + 2)
                    ax2.imshow(label_map_c, cmap='gray')
                    ax2.axis('off')

        mae = maes.avg
        mae_d = maes_d.avg

        mse = np.sqrt(mses.avg)
        mse_d = np.sqrt(mses_d.avg)

        loss = losses.avg
        loss_d = d_loss.avg


        self.writer.add_figure('density_map', fig, global_step=self.epoch + 1)

        self.writer.add_scalar(mode + '_val_loss', loss1, self.epoch + 1)
        self.writer.add_scalar(mode + '_val_loss_d', loss_d, self.epoch + 1)

        self.writer.add_scalar(mode + '_val_mae', mae1, self.epoch + 1)
        self.writer.add_scalar(mode + '_val_mae_d', mae_d, self.epoch + 1)

        self.writer.add_scalar(mode + '_val_mse', mse1, self.epoch + 1)
        self.writer.add_scalar(mode + '_val_mse_d', mse_d, self.epoch + 1)

        self.update_model([mae, mse, loss], mae_d=mae_d)

        print('{} Evaluation, Epoch: {}'.format(mode, self.epoch))
        print('\t\t val_loss: {:.6f}\t mae: {:.6f}\t  mse: {:.6f} \t '.format(
                loss, mae, mse))
        print('\t\t val_loss_d: {:.6f}\t mae_d: {:.6f}\t  mse_d: {:.6f} \t '.format(
                loss_d, mae_d, mse_d))

    def update_model(self, scores, mae_d, log_file=None):

        mae, mse, loss = scores

        snapshot_name = 'all_ep_%d_mae_%.1f_mse_%.1f_loss_%.1f' % (self.epoch + 1, mae, mse, loss)

        if mae < self.train_record['best_mae'] or mse < self.train_record['best_mse']:
            self.train_record['best_model_name'] = snapshot_name
            to_saved_weight = {'net': self.net.state_dict(), 'decoder': self.decoder.state_dict()}
            torch.save(to_saved_weight, osp.join(self.exp_path, self.exp_name, snapshot_name + '.pth'))

        if mae < self.train_record['best_mae']:           
            self.train_record['best_mae'] = mae

        if mse < self.train_record['best_mse']:
            self.train_record['best_mse'] = mse 

        # save best regression ckpt
        if mae_d < self.train_record['best_mae_d']:
            self.train_record['best_mae_d'] = mae_d
            snapshot_name = 'all_ep_%d_mae_d_%.1f' % (self.epoch + 1, mae_d)
            to_saved_weight = {'net': self.net.state_dict(), 'decoder': self.decoder.state_dict()}
            torch.save(to_saved_weight, osp.join(self.exp_path, self.exp_name, snapshot_name + '.pth'))

        latest_state = {'train_record': self.train_record, 'net': self.net.state_dict(), 'decoder': self.decoder.state_dict(), \
                        'optimizer_g': self.optimizer_g.state_dict(), 'optimizer_c': self.optimizer_c.state_dict(), \
                        'optimizer_d': self.optimizer_d.state_dict(), \
                        'epoch': self.epoch, 'i_tb': self.i_tb, 'exp_path': self.exp_path, \
                        'exp_name': self.exp_name}

        torch.save(latest_state, osp.join(self.exp_path, self.exp_name, 'latest_state.pth'))

