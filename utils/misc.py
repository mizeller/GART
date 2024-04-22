import torch
import numpy as np
import logging, platform


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def fov2K(fov=90, H=256, W=256):
    if isinstance(fov, torch.Tensor):
        f = H / (2 * torch.tan(fov / 2 * np.pi / 180))
        K = torch.eye(3).repeat(fov.shape[0], 1, 1).to(fov)
        K[:, 0, 0], K[:, 0, 2] = f, W / 2.0
        K[:, 1, 1], K[:, 1, 2] = f, H / 2.0
        return K.clone()
    else:
        f = H / (2 * np.tan(fov / 2 * np.pi / 180))
        K = np.eye(3)
        K[0, 0], K[0, 2] = f, W / 2.0
        K[1, 1], K[1, 2] = f, H / 2.0
        return K.copy()


def get_bbox(mask, padding):
    # mask: H,W, 0-1, get the bbox with padding
    assert mask.ndim == 2
    assert isinstance(mask, torch.Tensor)
    # x is width, y is hight
    xm = mask.sum(dim=0) > 0
    ym = mask.sum(dim=1) > 0

    xl, xr = xm.nonzero().min(), xm.nonzero().max()
    yl, yr = ym.nonzero().min(), ym.nonzero().max()

    xl, xr = max(0, xl - padding), min(mask.shape[1], xr + padding)
    yl, yr = max(0, yl - padding), min(mask.shape[0], yr + padding)

    box = torch.zeros_like(mask)
    box[yl:yr, xl:xr] = 1.0

    return yl, yr, xl, xr


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True
