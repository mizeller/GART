import torch, numpy as np
from tqdm import tqdm
from transforms3d.euler import euler2mat
from transforms3d.axangles import axangle2mat, mat2axangle
from sklearn.neighbors import KernelDensity
from torch import nn
from matplotlib import pyplot as plt


class RealDataOptimizablePoseProviderPose(nn.Module):
    # separate the base_R and rest_R
    def __init__(self, dataset, balance=True):
        super().__init__()
        self.dataset = dataset
        self.balance = balance
        (
            rgb_list,
            mask_list,
            K_list,
            pose_list,
            global_trans_list,
            betas,
        ) = self.prepare_for_fitting(dataset)
        for name, tensor in zip(
            ["rgb_list", "mask_list", "K_list", "betas"],
            [rgb_list, mask_list, K_list, betas],
        ):
            # self.register_buffer(name, tensor)
            # * don't register buffer, just save them in RAM
            setattr(self, name, tensor)

        self.pose_base_list = nn.Parameter(pose_list[:, :1])
        self.pose_rest_list = nn.Parameter(pose_list[:, 1:])
        self.global_trans_list = nn.Parameter(global_trans_list)
        self.register_buffer("pose_list_original", pose_list.clone())
        self.register_buffer("global_trans_list_original", global_trans_list.clone())

        return

    def move_images_to_device(self, device):
        self.rgb_list = self.rgb_list.to(device)
        self.mask_list = self.mask_list.to(device)
        self.K_list = self.K_list.to(device)
        self.betas = self.betas.to(device)

    # TODO: remove laters
    # @property
    # def total_t(self):
    #     return len(self.rgb_list)

    @property
    def pose_diff(self):
        optim_pose = self.pose_list
        ori_pose = self.pose_list_original
        pose_diff = optim_pose - ori_pose
        optim_trans = self.global_trans_list
        ori_trans = self.global_trans_list_original
        trans_diff = optim_trans - ori_trans
        return pose_diff, trans_diff

    def forward(
        self,
        N=None,
        continuous=False,
        index=None,
        return_index=False,
        force_uniform=False,
    ):
        device = self.pose_base_list.device
        # for uniform is for pose optimization
        # TODO: support multi-clips
        if index is not None:
            if not isinstance(index, torch.Tensor):
                index = torch.from_numpy(np.asarray(index)).long()
                if index.dim() == 0:
                    index = index[None]
            t = index
        else:
            assert N is not None
            if N == len(self.rgb_list) and not self.balance:
                t = torch.arange(N).long()
            else:
                if continuous and N > 1:
                    t0 = torch.randint(
                        low=0, high=len(self.rgb_list) - N, size=(1,)
                    ).squeeze()
                    t = torch.arange(t0, t0 + N).long()
                else:
                    prob = (
                        self.selection_prob
                        if not force_uniform
                        else (
                            np.ones_like(self.selection_prob) / len(self.selection_prob)
                        )
                    )
                    t = torch.from_numpy(
                        np.random.choice(len(self.rgb_list), N, p=prob)
                    ).long()
        if return_index:
            return t
        gt_rgb = self.rgb_list[t]  # B,H,W,3
        gt_mask = self.mask_list[t]  # B,H,W
        K = self.K_list[t]  # B,3,3
        pose_base = self.pose_base_list[t]
        pose_rest = self.pose_rest_list[t]
        global_trans = self.global_trans_list[t]

        return (
            gt_rgb.to(device),
            gt_mask.to(device),
            K.to(device),
            pose_base,
            pose_rest,
            global_trans,
            t.to(device),
        )

    def prepare_for_fitting(self, dataset):
        rgb_list, mask_list, K_list, pose_list, global_trans_list = [], [], [], [], []
        for t in tqdm(range(len(dataset))):
            poses, meta = dataset[t]
            betas, pose, global_trans = (
                poses["smpl_beta"],
                poses["smpl_pose"],
                poses["smpl_trans"],
            )
            betas, pose, global_trans = (
                torch.from_numpy(betas),
                torch.from_numpy(pose),
                torch.from_numpy(global_trans),
            )
            pose_list.append(pose), global_trans_list.append(global_trans)
            K = torch.from_numpy(poses["K"])
            gt_rgb = torch.from_numpy(poses["rgb"])
            mask = torch.from_numpy(poses["mask"])
            gt_rgb = gt_rgb * mask.unsqueeze(-1) + (1 - mask.unsqueeze(-1))
            # ! white background
            self.H, self.W = gt_rgb.shape[:2]  # H,W,3
            rgb_list.append(gt_rgb), K_list.append(K)
            mask_list.append(mask)
        pose_list = torch.stack(pose_list)  # T, 24, 3
        global_trans_list = torch.stack(global_trans_list)
        rgb_list = torch.stack(rgb_list)  # T, H, W, 3
        mask_list = torch.stack(mask_list)  # T, H, W
        K_list = torch.stack(K_list)  # T, 3, 3

        if not self.balance:
            self.selection_prob = np.ones(rgb_list.shape[0]) / rgb_list.shape[0]
            return rgb_list, mask_list, K_list, pose_list, global_trans_list, betas

        # * Weight the views
        global_rot = [p[0].detach().cpu().numpy() for p in pose_list]
        rot_list = []
        for w in tqdm(global_rot):
            angle = np.linalg.norm(w)
            dir = w / (angle + 1e-6)
            mat = axangle2mat(dir, angle)
            rot_list.append(mat)
        # use first frame as eye
        angle_list = []
        BASE_R = np.asarray(euler2mat(np.pi, 0, 0, "sxyz"))
        for R in rot_list:
            dR = np.matmul(BASE_R.T, R)
            angle = mat2axangle(dR)[1]
            angle_list.append(angle)
        angle_list = np.array(angle_list)

        # angle_list = np.sort(angle_list)
        kde = KernelDensity(kernel="gaussian", bandwidth=np.pi / 36.0).fit(
            np.concatenate(
                [angle_list - 2 * np.pi, angle_list, angle_list + 2 * np.pi]
            )[:, None]
        )  # Pad
        log_dens = kde.score_samples(angle_list[:, None])
        viz_angle = np.linspace(-np.pi, np.pi, 1000)
        log_viz_dens = kde.score_samples(viz_angle[:, None])

        _view_prob = np.exp(log_dens)
        _view_prob = _view_prob / np.sum(_view_prob)
        selection_prob = 1.0 / _view_prob
        selection_prob = selection_prob / np.sum(selection_prob)
        # print(_view_prob)

        self.selection_prob = selection_prob

        # for viz purpose
        self.viz_angle = viz_angle
        self.angle_list = angle_list
        self.log_viz_dens = log_viz_dens

        return rgb_list, mask_list, K_list, pose_list, global_trans_list, betas

    def viz_selection_prob(self, save_fn):
        if not self.balance:
            return
        fig = plt.figure(figsize=(12, 2))
        plt.subplot(1, 3, 1)
        plt.hist(self.angle_list, bins=100)
        plt.title("View angle histogram")
        plt.subplot(1, 3, 2)
        plt.plot(self.viz_angle, np.exp(self.log_viz_dens))
        plt.title("angle density")
        plt.subplot(1, 3, 3)
        plt.plot(self.selection_prob), plt.title("Selection Prob")
        plt.savefig(save_fn)
        # plt.show()
        plt.close()
