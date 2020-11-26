import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime

from models.Student.ShuffleNetV2 import shufflenet_v2_x1_0 as ShuffleV2
from models.Teacher.DMCount import vgg19
from wrapper import wrapper
from losses.ot_loss import OT_Loss
from misc.utils import Save_Handle, AverageMeter, get_logger, print_config, adjust_loss_weight, save_density_map
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[2])
    gt_discretes = torch.stack(transposed_batch[3], 0)
    return images, points, st_sizes, gt_discretes


class Trainer(object):
    def __init__(self, args, device, dataloaders):
        self.args = args
        self.device = device
        self.dataloaders = dataloaders

    def setup(self):
        args = self.args
        sub_dir = 'KD-input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}-{}'.format(
            args.crop_size, args.wot, args.wtv, args.reg, args.num_of_iter_in_ot, args.norm_cood,
            datetime.now().strftime("%y%m%d%H%M"))

        self.save_dir = os.path.join('exp', sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.logger = get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        self.writer = SummaryWriter(self.save_dir)
        print_config(vars(args), self.logger)

        downsample_ratio = 8

        self.alpha = 0.5
        self.t_model = vgg19()
        self.t_model.to(self.device)
        in_out_channels = [(24, 128), (116, 256), (232, 512), (464, 512), (128, 128)]
        self.s_model = wrapper(ShuffleV2(), in_out_channels, mask=[0, 0, 1, 1, 1])
        self.s_model.to(self.device)
        self.optimizer = optim.Adam(self.s_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[100, 500, 1000], gamma=0.5)
        self.start_epoch = 0
        self.epoch = 0
        self.i_tb = 0
        if args.resume:
            self.logger.info('loading pretrained model from ' + args.resume)
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.s_model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.s_model.load_state_dict(torch.load(args.resume, self.device))
        else:
            self.logger.info('random initialization')

        if args.teacher_ckpt:
            self.logger.info('loading pretrained teacher model from ' + args.teacher_ckpt)
            self.t_model.load_state_dict(torch.load(args.teacher_ckpt, self.device))
        else:
            self.logger.info('random initialization')

        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot,
                               args.reg)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=5)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.epoch = epoch
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()
            self.scheduler.step()

    def train_eopch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_kd_loss = AverageMeter()
        epoch_feature_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_t_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.t_model.eval()
        self.s_model.train()  # Set model to training mode

        for step, (inputs, points, st_sizes, gt_discrete) in enumerate(self.dataloaders['train']):
            self.i_tb += 1
            inputs = inputs.to(self.device)
            gt_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)
            with torch.no_grad():
                t_features, t_outputs, t_outputs_normed = self.t_model(inputs, is_feature=True)

            with torch.set_grad_enabled(True):
                s_features, s_outputs, s_outputs_normed = self.s_model(inputs, is_feature=True)
                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(s_outputs_normed, s_outputs, points)
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(s_outputs.sum(1).sum(1).sum(1),
                                      torch.from_numpy(gt_count).float().to(self.device))
                epoch_count_loss.update(count_loss.item(), N)

                # Compute TV loss.
                gt_count_tensor = torch.from_numpy(gt_count).float().to(self.device).unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
                gt_discrete_normed = gt_discrete / (gt_count_tensor + 1e-6)
                tv_loss = (self.tv_loss(s_outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                    1) * torch.from_numpy(gt_count).float().to(self.device)).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)

                s_loss = ot_loss + count_loss + tv_loss

                # kd_loss = self.mse(t_outputs, s_outputs)

                feature_loss = self.s_model.feature_loss(s_features, t_features, loss_type='cd')

                target_weight = adjust_loss_weight(1, self.epoch, lam=1, is_target_loss=True)
                feature_weight = adjust_loss_weight(self.args.wfeature, self.epoch, lam=0.9, is_target_loss=False)

                loss = target_weight * s_loss + feature_weight * feature_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = torch.sum(s_outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gt_count

                t_pred_count = torch.sum(t_outputs.view(N, -1), dim=1).detach().cpu().numpy()
                t_pred_err = t_pred_count - gt_count
                epoch_loss.update(loss.item(), N)
                epoch_kd_loss.update(kd_loss.item(), N)
                epoch_feature_loss.update(feature_loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)
                epoch_t_mae.update(np.mean(abs(t_pred_err)), N)

                self.writer.add_scalar('ot_loss', ot_loss, self.i_tb)
                self.writer.add_scalar('count_loss', count_loss, self.i_tb)
                self.writer.add_scalar('tv_loss', tv_loss, self.i_tb)
                self.writer.add_scalar('ot_obj_value', ot_obj_value, self.i_tb)
                self.writer.add_scalar('wd', wd, self.i_tb)
                self.writer.add_scalar('loss', loss, self.i_tb)
                self.writer.add_scalar('kd_loss', kd_loss, self.i_tb)
                self.writer.add_scalar('feature_loss', feature_loss, self.i_tb)


        self.logger.info(
            'Epoch {} Train, Loss: {:.4f}, KD_Loss: {:.4f}, feature_loss: {:.4f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
            'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, T_MAE: {:.2f}, Cost {:.1f} sec'
                .format(self.epoch, epoch_loss.get_avg(), epoch_kd_loss.get_avg(), epoch_feature_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_wd.get_avg(),
                        epoch_ot_obj_value.get_avg(), epoch_count_loss.get_avg(), epoch_tv_loss.get_avg(),
                        np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(), epoch_t_mae.get_avg(),
                        time.time() - epoch_start))
        model_state_dic = self.s_model.backbone.state_dict()
        model_state_dic_with_proj = self.s_model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic,
            'model_state_dic_with_proj': model_state_dic_with_proj
        }, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.s_model.eval()  # Set model to evaluate mode
        epoch_res = []
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                features, outputs, _ = self.s_model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.writer.add_scalar('val_mae', mae, self.epoch)
        self.writer.add_scalar('val_mse', mse, self.epoch)
        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.s_model.backbone.state_dict()
        model_state_dic_with_proj = self.s_model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            torch.save(model_state_dic_with_proj, os.path.join(self.save_dir, 'best_model_w_proj_{}.pth'.format(self.best_count)))
            self.best_count += 1

    def test(self, model):
        args = self.args
        epoch_start = time.time()
        model.eval()  # Set model to evaluate mode
        epoch_res = []
        for inputs, count, name in self.dataloaders['test']:
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                features, outputs, _ = model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

                utils.save_density_map(outputs.cpu().numpy()[0], 'outputs', name.split('.')[0] + '_output_' + '%d' % pred_count + '.png')

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.info('Test, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(mse, mae, time.time() - epoch_start))