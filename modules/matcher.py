import torch.nn as nn
from modules.blocks import convbn_3d

class BasicMatcher(nn.Module):
    def __init__(self, n_gwc_groups, n_concat_features):
        super(BasicMatcher, self).__init__()
        self.n_gwc_groups = n_gwc_groups
        self.n_concat_features = n_concat_features

        self.dres0 = nn.Sequential(convbn_3d(self.n_gwc_groups + self.n_concat_features * 2, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, kernel_size=3, stride=1, padding=1))
        self.dres2 = nn.Sequential(convbn_3d(32, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, kernel_size=3, stride=1, padding=1))
        self.classify = nn.Sequential(convbn_3d(32, 32, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, cost):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost = self.classify(cost0)

        return cost