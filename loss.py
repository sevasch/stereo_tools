import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, disparity_pred, disparity_gt):
        mask = (disparity_gt > 0).detach()  # only consider pixels with ground truth
        loss = F.smooth_l1_loss(disparity_pred, disparity_gt, reduction='none')
        loss *= mask
        loss = torch.mean(loss.sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1))
        return loss