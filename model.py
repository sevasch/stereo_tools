from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from modules.feature_extractor import FeatureExtractor
from modules.cost_volume import ConcatVolume, GwcVolume
from modules.matcher import BasicMatcher
from modules.aggregator import SoftmaxAggregator


class SimpleStereo(nn.Module):
    def __init__(self, cv_bins, n_concat_features=0, n_gwc_groups=0):
        super(SimpleStereo, self).__init__()

        # setup submodules
        self.feature_extractor = FeatureExtractor(use_gwc=(n_gwc_groups > 0), use_concat=(n_concat_features > 0), n_concat_features=n_concat_features)
        self.concat_volume = ConcatVolume(cv_bins)
        self.gwc_volume = GwcVolume(cv_bins, n_gwc_groups)
        self.matcher = BasicMatcher(n_gwc_groups, n_concat_features)
        self.aggregator = SoftmaxAggregator(cv_bins)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def update_bins(self, cv_bins_new):
        self.cv.update_bins(cv_bins_new)
        if hasattr(self.aggregator, 'bins'):
            self.aggregator.update_bins(cv_bins_new)
        elif hasattr(self.aggregator, 'D'):
            exit()

    def forward(self, im_left, im_right):

        # 2d feature extraction
        concat_left, gwc_left = self.feature_extractor(im_left)
        concat_right, gwc_right = self.feature_extractor(im_right)

        # creating cost volumes
        concat_vol_lr, _ = self.concat_volume(concat_left, concat_right, build_lr=True, build_rl=False)
        gwc_vol_lr, _ = self.gwc_volume(gwc_left, gwc_right, build_lr=True, build_rl=False)
        cv_lr = torch.cat((gwc_vol_lr, concat_vol_lr), 1)

        # 3D CNN
        matched_lr = self.matcher(cv_lr)

        # aggregatoion
        disp_left = self.aggregator(matched_lr.squeeze(1)).unsqueeze(1)

        return F.interpolate(disp_left, scale_factor=4, mode='bilinear', align_corners=True) * 4

