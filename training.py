import argparse
import datetime
import wandb
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from loss import L1Loss
from model import SimpleStereo
from utility.misc import tensor2numpy
import torch.multiprocessing as mp
from matplotlib import cm
from dataset import StereoDataset

''' ARGUMENT PARSER '''
parser = argparse.ArgumentParser(description='Train model. ')

# naming arguments
parser.add_argument('--description', default='SimpleStereo', help='additional description for the run')
parser.add_argument('--project', default='simple_stereo', help='project for logging')

# data arguments
parser.add_argument('--data_dir', default='/home/sebastian/Desktop/SF', help='data directory')
parser.add_argument('--max_disp', type=int, default=256, help='maximum disparity in the original image in pixels (assumption)')
parser.add_argument('--random_crop', type=int, nargs=2, default=[0, 0], help='if larger than 0, randomly crops to size (H, W)')
parser.add_argument('--center_crop', type=int, nargs=2, default=[0, 0], help='if larger than 0, crops to size (H, W)')  # [768, 960]
parser.add_argument('--batch_size_train', type=int, default=6, help='training batch size per GPU')
parser.add_argument('--batch_size_val', type=int, default=6, help='validation batch size per GPU')
parser.add_argument('--n_samples_train', type=int, default=1000, help='enter number >0 to limit the number of training samples')
parser.add_argument('--n_samples_val', type=int, default=20, help='enter number >0 to limit the number of validation samples')

# device specific
parser.add_argument('--world_size', type=int, default=1, help='number of GPUS')

# model arguments
parser.add_argument('--model_scale', type=float, default=1., help='scale of the model')
parser.add_argument('--n_layers', type=int, default=1, help='number of times data is run through LSTM')

# training arguments
parser.add_argument('--n_iterations', type=int, default=200, help='number of training epochs')
parser.add_argument('--steps_per_iteration', type=int, default=50, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.1, help='factor for learning rate decay, none if 0')
parser.add_argument('--lr_milestones', type=int, nargs='+', default=[50, 100], help='number of epochs after which lr is reduced by args.lr-decay')

# logging
parser.add_argument('--wandb_key', type=str, default=open('wandbkey.txt').read() if os.path.exists('wandbkey.txt') else '', help='if using wandb, insert key here')


''' FUNCTIONS '''
def train_step(args, sample, model, loss_fn, optimizer, scaler, rank=0):
    model.train()

    for key in ['im_left_t', 'im_right_t', 'im_left', 'im_right', 'disp_left', 'disp_right']:
        sample[key] = sample[key].cuda()

    # FORWARD PASS
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        disp_left_pred = model(sample['im_left_t'], sample['im_right_t'])
        disp_left_pred = F.interpolate(disp_left_pred, scale_factor=1/args.model_scale, mode='bilinear', align_corners=True) /args.model_scale
        loss = loss_fn(disp_left_pred, sample['disp_left'])

    # BACKWARD PASS
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # LOGGING
    if args.wandb_key and rank == 0:
        #TODO: move to eval
        rows = []
        im_left_log = sample['im_left'][0].detach().cpu().permute(1, 2, 0)
        disp_left_gt_log = cm.gist_ncar(tensor2numpy(sample['disp_left'])[0] / 255)[:, :, :3]
        disp_left_pred_log = cm.gist_ncar(tensor2numpy(disp_left_pred)[0] / 255)[:, :, :3]
        rows.append(np.concatenate((im_left_log, disp_left_gt_log, disp_left_pred_log), axis=1))
        all_images = np.concatenate(rows, axis=0)
        wandb.log({'example': wandb.Image(all_images)}, commit=False)

    return loss.detach().cpu()


def evaluate(args, dataloader, model, loss_fn):
    model.eval()

    losses = []
    for i, sample in enumerate(dataloader):
        for key in ['im_left_t', 'im_right_t', 'im_left', 'im_right', 'disp_left', 'disp_right']:
            sample[key] = sample[key].cuda()

        # FORWARD PASS
        with torch.cuda.amp.autocast(), torch.no_grad():
            disp_left_pred = model(sample['im_left_t'], sample['im_right_t'])
            disp_left_pred = F.interpolate(disp_left_pred, scale_factor=1 / args.model_scale, mode='bilinear', align_corners=True) / args.model_scale
            loss = loss_fn(disp_left_pred, sample['disp_left'])

        losses.append(loss.detach().cpu())
    return torch.tensor(losses).mean().item()

def worker(rank, args):
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    dist.init_process_group('nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # DATASETS
    if 'train' and 'val' in os.listdir(args.data_dir):
        subset_train = StereoDataset(os.path.join(args.data_dir, 'train'), max_disp=args.max_disp, model_image_scale=args.model_scale, random_crop=args.random_crop, center_crop=args.center_crop)
        subset_val = StereoDataset(os.path.join(args.data_dir, 'val'), max_disp=args.max_disp, model_image_scale=args.model_scale, random_crop=args.random_crop, center_crop=args.center_crop)
    else:
        dataset = StereoDataset(os.path.join(args.data_dir), max_disp=args.max_disp, model_image_scale=args.model_scale, random_crop=args.random_crop, center_crop=args.center_crop)
        subset_train, subset_val = torch.utils.data.random_split(dataset, lengths=[int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    if args.n_samples_train:
        subset_train, _ = torch.utils.data.random_split(subset_train, [args.n_samples_train, len(subset_train) - args.n_samples_train])
    if args.n_samples_val:
        subset_val, _ = torch.utils.data.random_split(subset_val, [args.n_samples_val, len(subset_val) - args.n_samples_val])

    print('training set: ' + str(len(subset_train)))
    print('validation set: ' + str(len(subset_val)))

    # CREATE DATALOADER
    sampler_train = torch.utils.data.distributed.DistributedSampler(subset_train, num_replicas=args.world_size, rank=rank)
    sampler_val = torch.utils.data.distributed.DistributedSampler(subset_val, num_replicas=args.world_size, rank=rank)
    loader_train = DataLoader(subset_train, batch_size=args.batch_size_train, num_workers=3, sampler=sampler_train, pin_memory=True)
    loader_val = DataLoader(subset_val, batch_size=args.batch_size_val, num_workers=3, sampler=sampler_val, pin_memory=True)

    # CREATE MODEL
    cv_bins = torch.arange(0, args.model_scale * args.max_disp / 4, 4).type(torch.int)
    model = SimpleStereo(cv_bins, n_concat_features=8, n_gwc_groups=4)
    model.cuda()

    # LOSS FUNCTION
    loss_fn = L1Loss()

    # OPTIMIZER AND LR_SCHEDULER
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # , eps=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_decay) if args.lr_decay else None

    # MIXED PRECISION SETUP
    scaler = torch.cuda.amp.GradScaler()

    # MULTI GPU SETUP
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank], find_unused_parameters=True)

    # LOGGING SETUP
    if rank == 0 and args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.project, name=args.run_name, config=args)
        wandb.watch(model)

    print('started training on rank ' + str(rank))

    # TRAINING
    iterator_train = iter(loader_train)
    sample_no = 0  # keeps track of sample in loader to reshuffle
    epoch = 0
    losses_train = []
    losses_val = []
    sampler_train.set_epoch(epoch)
    for iteration in range(args.n_iterations):
        # TRAINING
        loss_train = []
        for step in range(args.steps_per_iteration):
            sample = next(iterator_train)  # grab next batch
            loss = train_step(args, sample, model, loss_fn, optimizer, scaler, rank)
            loss_train.append(loss)
            print('rank ' + str(rank) + ', iteration ' + str(iteration + 1) + '/' + str(args.n_iterations) + ', train step ' + str(step + 1) + '/' + str(args.steps_per_iteration) + ', finished, loss: ' + str(loss.item()))

            # check if samples left in dataloader, otherwise reshuffle
            sample_no += 1
            if sample_no >= len(loader_train):
                sample_no = 0
                epoch += 1
                print('rank ' + str(rank) + ', reshuffle for epoch: ' + str(epoch))
                sampler_train.set_epoch(epoch)
                iterator_train = iter(loader_train)  # iter -> triggers __iter__() -> shuffles data


        # VALIDATION
        sampler_val.set_epoch(iteration)
        loss_val = evaluate(args, loader_val, model, loss_fn)
        print('rank ' + str(rank) + ', iteration ' + str(iteration + 1) +  '/' + str(args.n_iterations) + ', evaluation loss: ' +  str(loss_val))

        # LEARNING RATE UPDATE
        if lr_scheduler:
            lr_scheduler.step()

        # LOGGING
        losses_train.append(torch.tensor(loss_train).mean())
        losses_val.append(loss_val)

        if args.wandb_key and rank == 0:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            wandb.log({'loss_t': torch.tensor(loss_train).mean().item(),
                       'loss_v': loss_val,
                       'learning_rate': lr}, commit=True)

        # SAVE MODEL CHECKPOINT
        if rank == 0:
            if loss_val == min(losses_val):
                ckpt = {'model_state_dict': model.module.state_dict()}
                torch.save(ckpt, os.path.join(args.run_dir, 'iter' + str(iteration + 1).zfill(2) + ('_best.pt')))


''' MAIN '''
if __name__ == '__main__':
    args = parser.parse_args()

    # NAMING
    now = datetime.datetime.now()
    timestamp = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '_' \
                + str(now.hour).zfill(2) + str(now.minute).zfill(2)
    args.run_name = timestamp + '_' + args.description
    args.run_dir = os.path.join('saved_models', args.run_name)
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    # SAVE CONFIG
    json.dump(args.__dict__, open(os.path.join(args.run_dir, 'args.json'), 'w'))

    # START PROCESSES
    mp.spawn(worker, nprocs=args.world_size, args=(args,))

