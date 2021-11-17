import numpy as np

def D1(disp_pred, disp_gt):
    disp_pred = disp_pred * (disp_gt > 0)  # only evaluate values where there exists ground truth
    error_abs = np.abs(disp_gt - disp_pred)
    error_rel = error_abs / np.maximum(disp_gt, np.ones_like(disp_gt))  # avoid diviision by 0

    # calculate masks (>3px AND >0.05*disp)
    mask = (error_abs > 3) * (error_rel > 0.05)  # mask contains pixels ABOVE both thresholds
    D1 = 100. * mask.sum() / np.max((np.count_nonzero(disp_gt), 1))  # avoid division by 0

    return D1

def pc_error(pc_pred, pc_gt):
    # count pixels
    idx = np.where(pc_gt.sum(axis=-1) > 0)  # only where GT is available
    n_pixels = pc_gt.shape[0] * pc_gt.shape[1]  # number of image pixels
    n_pixels_valid = len(idx[0])  # number of pixels with gt disparity

    # calculate metrics
    L1_error = np.nan
    L2_error = np.nan
    depth_error = np.nan

    if n_pixels_valid / n_pixels >= 0.1:
        L1_error = np.mean(np.abs(pc_gt[idx] - pc_pred[idx]))
        L2_error = np.mean(np.linalg.norm(pc_gt[idx] - pc_pred[idx], axis=1))
        depth_error = np.mean(np.abs(pc_gt[idx][:, 2] - pc_pred[idx][:, 2]))

    return L1_error, L2_error, depth_error

