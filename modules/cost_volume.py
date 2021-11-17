import torch.nn as nn

class ConcatVolume(nn.Module):
    def __init__(self, bins):
        super(ConcatVolume, self).__init__()
        self.update_bins(bins)

        print('concat volume q: ' + str(self.bins))

    def update_bins(self, new_bins):
        self.D = new_bins.shape[0]
        self.bins = new_bins

    def _build_concat_volume(self, features_left, features_right, build_lr=True, build_rl=True):
        B, F, H, W = features_left.shape

        concat_vol_lr = features_left.new_zeros([B, 2 * F, self.bins.shape[0], H, W]) if build_lr else None
        concat_vol_rl = features_right.new_zeros([B, 2 * F, self.bins.shape[0], H, W]) if build_rl else None
        for i, d in enumerate(self.bins):
            if build_lr:
                if d > 0:
                    concat_vol_lr[:, :F, i, :, d:] = features_left[:, :, :, d:]
                    concat_vol_lr[:, F:, i, :, d:] = features_right[:, :, :, :-d]
                else:
                    concat_vol_lr[:, :F, i, :, :] = features_left
                    concat_vol_lr[:, F:, i, :, :] = features_right
            if build_rl:
                if d > 0:
                    concat_vol_rl[:, :F, i, :, :-d] = features_left[:, :, :, d:]
                    concat_vol_rl[:, F:, i, :, :-d] = features_right[:, :, :, :-d]
                else:
                    concat_vol_rl[:, :F, i, :, :] = features_left
                    concat_vol_rl[:, F:, i, :, :] = features_right
        concat_vol_lr = concat_vol_lr.contiguous() if build_lr else None
        concat_vol_rl = concat_vol_rl.contiguous() if build_rl else None
        return concat_vol_lr, concat_vol_rl


    def forward(self, features_left, features_right, build_lr=True, build_rl=True):
        return self._build_concat_volume(features_left, features_right, build_lr, build_rl)


class GwcVolume(nn.Module):
    def __init__(self, bins, n_gwc_groups=0):
        super(GwcVolume, self).__init__()
        self.update_bins(bins)
        self.n_gwc_groups = n_gwc_groups

        print('gwc volume q: ' + str(self.bins))

    def update_bins(self, new_bins):
        self.D = new_bins.shape[0]
        self.bins = new_bins

    def _groupwise_correlation(self, fea1, fea2, n_gwc_groups):
        B, F, H, W = fea1.shape
        assert F % n_gwc_groups == 0
        channels_per_group = F // n_gwc_groups
        cost = (fea1 * fea2).view([B, n_gwc_groups, channels_per_group, H, W]).mean(dim=2)
        assert cost.shape == (B, n_gwc_groups, H, W)
        return cost

    def _build_gwc_volume(self, features_left, features_right, build_lr=True, build_rl=True):
        B, F, H, W = features_left.shape
        gwc_vol_lr = features_left.new_zeros([B, self.n_gwc_groups, self.bins.shape[0], H, W]) if build_lr else None
        gwc_vol_rl = features_right.new_zeros([B, self.n_gwc_groups, self.bins.shape[0], H, W]) if build_rl else None
        for i, d in enumerate(self.bins):
            if build_lr:
                if d > 0:
                    gwc_vol_lr[:, :, i, :, d:] = self._groupwise_correlation(features_left[:, :, :, d:], features_right[:, :, :, :-d], self.n_gwc_groups)
                else:
                    gwc_vol_lr[:, :, i, :, :] = self._groupwise_correlation(features_left, features_right, self.n_gwc_groups)
            if build_rl:
                if d > 0:
                    gwc_vol_rl[:, :, i, :, :-d] = self._groupwise_correlation(features_left[:, :, :, d:], features_right[:, :, :, :-d], self.n_gwc_groups)
                else:
                    gwc_vol_rl[:, :, i, :, :] = self._groupwise_correlation(features_left, features_right, self.n_gwc_groups)
        gwc_vol_lr = gwc_vol_lr.contiguous() if build_lr else None
        gwc_vol_rl = gwc_vol_rl.contiguous() if build_rl else None
        return gwc_vol_lr, gwc_vol_rl

    def forward(self, features_left, features_right, build_lr=True, build_rl=True):
        return self._build_gwc_volume(features_left, features_right, build_lr, build_rl)
