import torch.nn as nn

def convbn_2d(in_features, out_features, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else padding, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_features))

def convbn_3d(in_features, out_features, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv3d(in_features, out_features, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
                         nn.BatchNorm3d(out_features))