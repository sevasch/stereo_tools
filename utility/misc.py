import numpy as np


def tensor2numpy(input):
    if input.ndim == 3:
        output = input.detach().squeeze().cpu().numpy()
    elif input.ndim == 4:
        output = input.detach().permute(0, -2, -1, -3).squeeze(-1).cpu().numpy()
    elif input.ndim == 5:
        output = input.detach().permute(0, 1, -2, -1, -3).squeeze(-1).cpu().numpy()
    return output


def extract_baseline(calibration_data):
    return float(np.abs(calibration_data['ex_T'][..., 0, 0]))


def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def check_alignment(image, disp, width=100):
    disp = np.moveaxis(np.tile(disp, (3, 1, 1)), 0, -1)
    out = np.copy(image)
    for i in np.arange(0, out.shape[1], width):
        out[:, i:i+width//2] = disp[:, i:i+width//2]*2
    return out


def remove_module(state_dict):
    state_dict_new = {k[7:]: v for k, v in state_dict.items()}  # remove 'module.'
    return state_dict_new

