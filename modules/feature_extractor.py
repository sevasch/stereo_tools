import torch
import torch.nn as nn
from modules.blocks import convbn_2d

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1, downsample=None, padding=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn_2d(in_features, out_features, kernel_size=3, stride=stride, padding=padding, dilation=dilation),
                                   nn.ReLU(inplace=True))
        self.conv2 = convbn_2d(out_features, out_features, kernel_size=3, stride=1, padding=padding, dilation=dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class FeatureExtractor(nn.Module):
    def __init__(self, use_gwc=False, use_concat=False, n_concat_features=0):
        super(FeatureExtractor, self).__init__()
        self.use_gwc = use_gwc
        self.use_concat = use_concat
        self.in_features = 16

        self.conv0_1 = nn.Sequential(convbn_2d(3, 16, kernel_size=3, stride=2, padding=1, dilation=1),
                                     nn.ReLU(inplace=True))
        self.conv0_2 = nn.Sequential(convbn_2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1),
                                     nn.ReLU(inplace=True))
        self.conv0_3 = nn.Sequential(convbn_2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1),
                                     nn.ReLU(inplace=True))

        self.conv1_x = self._make_layer(out_features=16, num_blocks=3, stride=1, padding=1, dilation=1)
        self.conv2_x = self._make_layer(out_features=16, num_blocks=8, stride=2, padding=1, dilation=1)
        self.conv3_x = self._make_layer(out_features=32, num_blocks=3, stride=1, padding=1, dilation=1)
        self.conv4_x = self._make_layer(out_features=32, num_blocks=3, stride=1, padding=1, dilation=2)

        if use_concat:
            self.fusion = nn.Sequential(convbn_2d(80, 32, kernel_size=3, stride=1, padding=1, dilation=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, n_concat_features, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, out_features, num_blocks, stride, padding, dilation):
        downsample = None
        if stride != 1 or self.in_features != out_features:
            downsample = nn.Sequential(nn.Conv2d(self.in_features, out_features, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_features))

        layers = []
        layers.append(ResidualBlock(self.in_features, out_features, stride=stride, downsample=downsample, padding=padding, dilation=dilation))
        self.in_features = out_features
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(self.in_features, out_features, stride=1, downsample=None, padding=padding, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        conv0_1_out = self.conv0_1(x)
        conv0_2_out = self.conv0_2(conv0_1_out)
        conv0_3_out = self.conv0_3(conv0_2_out)

        conv1_x_out = self.conv1_x(conv0_3_out)
        conv2_x_out = self.conv2_x(conv1_x_out)
        conv3_x_out = self.conv3_x(conv2_x_out)
        conv4_x_out = self.conv4_x(conv3_x_out)

        gwc_features = torch.cat((conv2_x_out, conv3_x_out, conv4_x_out), dim=1)

        if self.use_concat and self.use_gwc:
            return self.fusion(gwc_features), gwc_features
        if self.use_concat:
            return self.fusion(gwc_features), None
        if self.use_gwc:
            return None, gwc_features


if __name__ == '__main__':
    import time
    fe = FeatureExtractor(use_gwc=True, use_concat=True, n_concat_features=32)
    fe.cuda()
    im_test = torch.randn(1, 3, 512, 640).cuda()

    fe.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(20):
            concat, gwc = fe(im_test)
        print((time.time()-start)/20)
        print(concat.shape, gwc.shape)