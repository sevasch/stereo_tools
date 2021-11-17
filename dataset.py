import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utility.pfm import read_PFM

#TODO: move interpolation outside of dataset, maybe avoid _t

class StereoDataset(Dataset):
    def __init__(self, root_dir, max_disp=0, model_image_scale=1., random_crop=(0, 0), center_crop=(0, 0)):
        self.max_disp = max_disp  # maximum disparity value expected in data
        self.model_image_scale = model_image_scale  # if different from 1, returns scaled image with _t ending
        self.random_crop = random_crop
        self.center_crop = center_crop

        self.keys = [subdir for subdir in sorted(os.listdir(root_dir)) if subdir in ['im_left', 'im_right', 'disp_left', 'disp_right']]

        # get all filenames for each key
        self.filenames_per_key = {}
        for key in self.keys:
            filenames_to_add = [os.path.join(root_dir, key, filename) for filename in sorted(os.listdir(os.path.join(root_dir, key)))]
            self.filenames_per_key[key] = filenames_to_add
            assert len(filenames_to_add) == len(self.filenames_per_key[self.keys[0]])

    def __len__(self):
        return len(self.filenames_per_key[self.keys[0]])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        for key, filenames in self.filenames_per_key.items():
            if key == 'im_left' or key == 'im_right':
                image = cv2.cvtColor(cv2.imread(filenames[idx]), cv2.COLOR_BGR2RGB) / 255.
                sample[key] = torch.from_numpy(image.transpose([2, 0, 1])).type(torch.FloatTensor)

            elif key == 'disp_left' or key == 'disp_right':
                disp = np.abs(read_PFM(filenames[idx])).clip(0, self.max_disp)
                sample[key] = torch.from_numpy(disp.astype(np.float32)).type(torch.FloatTensor).unsqueeze(0)

        # store filename
        sample['filename'] = os.path.split(self.filenames_per_key[self.keys[0]][idx])[-1].split('.')[0]

        # concatenate all samples to single tensor and apply transforms
        for key in ['im_left', 'im_right', 'disp_left', 'disp_right']:
            if key in sample:
                # center crop
                if self.center_crop[0] > 0 and self.center_crop[1] > 0:
                    new_h, new_w = self.center_crop
                    _, _, h, w = sample[key].shape
                    top = (h - new_h) // 2
                    left = (w - new_w) // 2
                    if new_h < h and new_w < w:
                        sample[key] = sample[key][..., top: top + new_h, left: left + new_w]

                # random crop
                if self.random_crop[0] > 0 and self.random_crop[1] > 0:
                    new_h, new_w = self.random_crop
                    _, _, h, w = sample[key].shape
                    if new_h < h and new_w == w:
                        top = np.random.randint(0, h - new_h)
                        sample[key] = sample[key][..., top: top + new_h]
                    if new_h == h and new_w < w:
                        left = np.random.randint(0, w - new_w)
                        sample[key] = sample[key][..., left: left + new_w]
                    if new_h < h and new_w < w:
                        top = np.random.randint(0, h - new_h)
                        left = np.random.randint(0, w - new_w)
                        sample[key] = sample[key][..., top: top + new_h, left: left + new_w]

        # downscaled image for model
        for key in ['im_left', 'im_right']:
            sample[key + '_t'] = F.interpolate(sample[key].unsqueeze(0), scale_factor=self.model_image_scale, mode='bilinear', align_corners=False).squeeze(0)

        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    path = '/home/sebastian/Desktop/SF/val'
    ds = StereoDataset(path, max_disp=500)
    dl = DataLoader(ds)

    for sample in dl:
        # print(sample['im_left'].shape)
        print(sample['disp_right'].max())

        plt.imshow(sample['im_left'].squeeze().permute(1, 2, 0))
        plt.show()

