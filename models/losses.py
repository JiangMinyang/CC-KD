import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, gt):
        pos_inds = gt.gt(0.99).float()
        neg_inds = gt.lt(0.99).float()

        neg_weights = torch.pow(1 - gt, 2)

        pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds 
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = max(1, pos_inds.float().sum())
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        return -(pos_loss + neg_loss) / num_pos