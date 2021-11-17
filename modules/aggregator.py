import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxAggregator(nn.Module):
    def __init__(self, bins):
        super(SoftmaxAggregator, self).__init__()
        self.D = bins.shape[0]
        self.disp_values = bins.view(1, self.D, 1, 1).cuda()

    def update_bins(self, new_bins):
        self.D = new_bins.shape[0]
        self.disp_values = new_bins.view(1, self.D, 1, 1).cuda()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        out = torch.sum(x * self.disp_values, 1, keepdim=False)
        return out