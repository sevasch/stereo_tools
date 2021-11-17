import os
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from models import ModularStereo, LSTMWrapper, StereoLSTM2, FlowStereo
from models.flow_stereo.FlowNet.FlowNetS import FlowNetS
from torch.utils.data import DataLoader
from data.sequence_dataset import StereoSequence
from metrics import D1, pc_error
from utility.pointcloud import disp2pc
from utility.misc import tensor2numpy
from tqdm import tqdm

parser = argparse.ArgumentParser(description='evaluate model')
parser.add_argument('--data_dirs', nargs='+', default=['/media/sebastian/linux/datasets/SCARED_all_corr_testset/dataset_8/keyframe_1',
                                                       '/media/sebastian/linux/datasets/SCARED_all_corr_testset/dataset_8/keyframe_2',
                                                       '/media/sebastian/linux/datasets/SCARED_all_corr_testset/dataset_8/keyframe_3',
                                                       '/media/sebastian/linux/datasets/SCARED_all_corr_testset/dataset_8/keyframe_4',
                                                       '/media/sebastian/linux/datasets/SCARED_all_corr_testset/dataset_9/keyframe_1',
                                                       '/media/sebastian/linux/datasets/SCARED_all_corr_testset/dataset_9/keyframe_2',
                                                       '/media/sebastian/linux/datasets/SCARED_all_corr_testset/dataset_9/keyframe_3',
                                                       '/media/sebastian/linux/datasets/SCARED_all_corr_testset/dataset_9/keyframe_4', ], help='datasets to evaluate on')
# parser.add_argument('--data_dirs', nargs='+', default=['/media/sebastian/linux/datasets/SCARED_all_corr_testset/dataset_8/keyframe_1', ], help='dataset to evaluate on')
parser.add_argument('--max_disp', type=int, default=256, help='maximum disparity value of data')
parser.add_argument('--model_scale', type=float, default=0.5, help='scale of the data for the model')
parser.add_argument('--n_sequence', type=int, default=2, help='length of image sequence')
parser.add_argument('--n_step', type=int, default=5, help='number of images to step in between images in the sequence')
parser.add_argument('--n_skip', type=int, default=5, help='every n_skip sample is evaluated')
parser.add_argument('--save_file', default='performance_all_5th_SMfinetuned_.json', help='.json file to save results in')


''' FUNCTIONS '''
def find_best_epoch(dir):
    # find latest best checkpoint
    epochs = os.listdir(dir)
    best_epochs = []
    epochs.sort(key=lambda x: x.split('_')[-1])
    for epoch in epochs:
        if epoch.split('_')[-1] == 'best.pt':
            best_epochs.append(epoch)
    best_epochs.sort()
    return best_epochs[-1]


def evaluate(model, dataloader):
    errors = {key: [] for key in ['filename', 'D1', 'L1', 'L2', 'depth']}

    for sample_no, sample in enumerate(tqdm(dataloader)):
        if np.mod(sample_no, args.n_skip) == 0:
            # temporary fix for old calibration data of test datasets
            calibration_data = sample['calibration_data'][-1]
            C = np.array(calibration_data['C_left'][-1])
            # C = np.zeros((3, 3))
            # C[0, 0] = float(calibration_data['c_left_f'][..., 0])
            # C[1, 1] = float(calibration_data['c_left_f'][..., 1])
            # C[0, 2] = float(calibration_data['c_left_c'][..., 0])
            # C[1, 2] = float(calibration_data['c_left_c'][..., 1])
            baseline = float(np.abs(calibration_data['ex_T'][..., 0, 0]))

            # make prediction, resize disparity to full scale and calculate point cloud
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                disps_left_pred = model(sample['im_left_t'].cuda(), sample['im_right_t'].cuda())

            disp_left_pred = F.interpolate(disps_left_pred[:, -1], scale_factor=4/args.model_scale, mode='bilinear', align_corners=False) * 4/args.model_scale

            # use only first batch (requires batch_size=1)
            disp_left_pred = tensor2numpy(disp_left_pred)[0]   # take last output in case of SHG
            disp_left_gt = tensor2numpy(sample['disp_left'])[0, -1]
            pc_left_pred = disp2pc(disp_left_pred, C, baseline)
            pc_left_gt = tensor2numpy(sample['pc_left'])[0, -1]

            # store values
            D1_error = D1(disp_left_pred, disp_left_gt)
            L1_error, L2_error, depth_error = pc_error(pc_left_pred, pc_left_gt)
            errors['filename'].append(sample['filenames'][-1][0])
            errors['D1'].append(D1_error)
            errors['L1'].append(L1_error)
            errors['L2'].append(L2_error)
            errors['depth'].append(depth_error)

    return errors

''' MAIN '''
if __name__ == '__main__':
    torch.manual_seed(1)

    args = parser.parse_args()

    # LOAD MODEL
    cv_bins = torch.arange(0, args.model_scale * args.max_disp / 4, 1).type(torch.int)
    model = ModularStereo(cv_bins, n_concat_features=32, n_gwc_groups=8)
    ckpt = torch.load('/home/sebastian/PycharmProjects/stereodepth/saved_models/20201018_1748_stereomodel_finetune/iter17_best.pt')
    model.load_state_dict(ckpt['model_state_dict'])

    # flownet = FlowNetS(batchNorm=True)
    # cv_bins = torch.arange(0, args.model_scale * args.max_disp / 4, 1).type(torch.int)
    # model = FlowStereo(flownet, cv_bins, n_concat_features=32, n_gwc_groups=8)
    # # ckpt= torch.load('/home/ebastian/PycharmProjects/stereodepth/saved_models/20201021_0108_FlowStereo_1/iter42_best.pt')
    # ckpt= torch.load('/home/sebastian/PycharmProjects/stereodepth/saved_models/20201021_0126_FlowStereo_5/iter85_best.pt')
    # model.load_state_dict(ckpt['model_state_dict'])


    # EVALUATE
    all_results = {key: [] for key in ['filename', 'D1', 'L1', 'L2', 'depth']}
    for data_dir in sorted(args.data_dirs):
        print('evaluating on ' + os.path.basename(os.path.split(data_dir)[0]) + '_' + os.path.basename(data_dir))

        # load dataset
        dataset = StereoSequence(data_dir, sequence_length=args.n_sequence, step=args.n_step, max_disp=args.max_disp, model_image_scale=args.model_scale, center_crop=[2*384, 2*480])
        loader = DataLoader(dataset, batch_size=1, num_workers=20, shuffle=False)

        # evaluate and append results
        results = evaluate(model.cuda(), loader)
        all_results = {key: all_results[key] + results[key] for key in all_results.keys()}

    # save results
    with open(args.save_file, 'w') as file:
        json.dump(all_results, file)

