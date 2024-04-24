import numpy as np
import torch


def transform_mu_frame(mu, frame, T):
    if len(mu) != len(T):
        assert len(mu) == 1 and len(frame) == 1
        mu = mu.expand(len(T), -1, -1)
        frame = frame.expand(len(T), -1, -1, -1)
    R, t = T[:, :3, :3], T[:, :3, 3]
    new_frame = torch.einsum("bij, bnjk->bnik", R, frame)
    new_mu = torch.einsum("bij, bnj->bni", R, mu) + t[:, None]
    return new_mu, new_frame


def sph_order2nfeat(order):
    return (order + 1) ** 2


def get_predefined_human_rest_pose(pose_type):
    print(f"Using predefined pose: {pose_type}")
    body_pose_t = torch.zeros((1, 69))
    if pose_type.lower() == "da_pose":
        body_pose_t[:, 2] = np.pi / 6
        body_pose_t[:, 5] = -np.pi / 6
    elif pose_type.lower() == "a_pose":
        body_pose_t[:, 2] = 0.2
        body_pose_t[:, 5] = -0.2
        body_pose_t[:, 47] = -0.8
        body_pose_t[:, 50] = 0.8
    elif pose_type.lower() == "t_pose":
        pass
    else:
        raise ValueError("Unknown cano_pose: {}".format(pose_type))
    return body_pose_t.reshape(23, 3)
