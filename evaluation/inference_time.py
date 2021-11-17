import os
import csv
import numpy as np
import time
import torch
import argparse
from models import ModularStereo, LSTMWrapper, StereoLSTM2, FlowStereo
from models.flow_stereo.FlowNet.FlowNetS import FlowNetS

''' ARGUMENT PARSER '''
parser = argparse.ArgumentParser(description='model evaluation')

parser.add_argument('--description', default='sm', help='description of the test')
parser.add_argument('--save_file', default='prediction_times.txt', help='.txt file to save result in')
parser.add_argument('--dims', type=int, nargs=2, default=[512, 640], help='image dimensions')
parser.add_argument('--n', type=int, default=100, help='number of predictions to calculate mean prediction time')


''' LOCALS '''

''' FUNCTIONS '''
def get_time(model, dims, n):
    im_left = torch.zeros(1, 1, 3, dims[0], dims[1]).cuda()
    im_right = torch.zeros(1, 1, 3, dims[0], dims[1]).cuda()

    model.eval().cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        print('warmup...')
        for i in range(20):
            print(i)
            model(im_left, im_right)

        print('benchmarking...')
        start = time.time()
        for _ in range(n):
            model(im_left, im_right)
    return (time.time() - start)/n


''' MAIN '''
if __name__ == '__main__':
    args = parser.parse_args()

    cv_bins = torch.arange(0, 128 / 4, 1).type(torch.int)
    model = ModularStereo(cv_bins, n_concat_features=32, n_gwc_groups=8)
    # model.load_state_dict(torch.load('../models/modularstereo/checkpoints/pretrained_SF_finetuned_SCARED.pt'))

    # flownet = FlowNetS(batchNorm=True)
    # cv_bins = torch.arange(0, 128 / 4, 1).type(torch.int)
    # model = FlowStereo(flownet, cv_bins, n_concat_features=32, n_gwc_groups=8)
    # # ckpt= torch.load('/home/ebastian/PycharmProjects/stereodepth/saved_models/20201021_0108_FlowStereo_1/iter42_best.pt')
    # ckpt= torch.load('/home/sebastian/PycharmProjects/stereodepth/saved_models/20201021_0126_FlowStereo_5/iter85_best.pt')
    # model.load_state_dict(ckpt['model_state_dict'])

    with open(os.path.join(args.save_file), 'a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([args.description, np.round(get_time(model, args.dims, 100), 4)])
