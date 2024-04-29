from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from torch.utils.tensorboard import SummaryWriter
from transforms3d.euler import euler2mat
import os, os.path as osp, shutil
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm
import numpy as np
import imageio
import logging
import time
import random
import math
import torch
from torch import nn

# imports from GART library
from lib_gart.model_utils import get_predefined_human_rest_pose
from lib_gart.smplx.smplx.lbs import batch_rigid_transform
from lib_gart.voxel_deformer import VoxelDeformer
from lib_gart.model import GaussianTemplateModel
from lib_gart.templates import SMPLTemplate
from lib_gart.smplx.smplx import SMPLLayer
from lib_gart.optim_utils import *

# other imports from local files...
from utils.misc import seed_everything, HostnameFilter, get_bbox
from lib_render.gauspl_renderer import render_cam_pcl
from utils.viz_utils import viz_spinning
from utils.viz import viz_render
from utils.ssim import ssim

########################################
# previously in lib_gart/model.py
# A template only handle the query of the


class SMPLTemplate(nn.Module):
    def __init__(self, smpl_model_path, init_beta, cano_pose_type, voxel_deformer_res):
        super().__init__()
        self.dim = 24
        self._template_layer = SMPLLayer(model_path=smpl_model_path)

        if init_beta is None:
            init_beta = np.zeros(10)
        init_beta = torch.as_tensor(init_beta, dtype=torch.float32).cpu()
        self.register_buffer("init_beta", init_beta)
        self.cano_pose_type = cano_pose_type
        self.name = "smpl"

        can_pose = get_predefined_human_rest_pose(cano_pose_type)
        can_pose = axis_angle_to_matrix(torch.cat([torch.zeros(1, 3), can_pose], 0))
        self.register_buffer("canonical_pose", can_pose)

        init_smpl_output = self._template_layer(
            betas=init_beta[None],
            body_pose=can_pose[None, 1:],
            global_orient=can_pose[None, 0],
            return_full_pose=True,
        )
        J_canonical, A0 = init_smpl_output.J, init_smpl_output.A
        A0_inv = torch.inverse(A0)
        self.register_buffer("A0_inv", A0_inv[0])
        self.register_buffer("J_canonical", J_canonical)

        v_init = init_smpl_output.vertices  # 1,6890,3
        v_init = v_init[0]
        W_init = self._template_layer.lbs_weights  # 6890,24

        self.voxel_deformer = VoxelDeformer(
            vtx=v_init[None],
            vtx_features=W_init[None],
            resolution_dhw=[
                voxel_deformer_res // 4,
                voxel_deformer_res,
                voxel_deformer_res,
            ],
        )

        # * Important, record first joint position, because the global orientation is rotating using this joint position as center, so we can compute the action on later As
        j0_t = init_smpl_output.joints[0, 0]
        self.register_buffer("j0_t", j0_t)
        return

    def get_init_vf(self):
        init_smpl_output = self._template_layer(
            betas=self.init_beta[None],
            body_pose=self.canonical_pose[None, 1:],
            global_orient=self.canonical_pose[None, 0],
            return_full_pose=True,
        )
        v_init = init_smpl_output.vertices  # 1,6890,3
        v_init = v_init[0]
        faces = self._template_layer.faces_tensor
        return v_init, faces

    def forward(self, theta=None, xyz_canonical=None):
        # skinning
        if theta is None:
            A = None
        else:
            assert (
                theta.ndim == 3 and theta.shape[-1] == 3
            ), "pose should have shape Bx24x3, in axis-angle format"
            nB = len(theta)
            _, A = batch_rigid_transform(
                axis_angle_to_matrix(theta),
                self.J_canonical.expand(nB, -1, -1),
                self._template_layer.parents,
            )
            A = torch.einsum("bnij, njk->bnik", A, self.A0_inv)  # B,24,4,4

        if xyz_canonical is None:
            # forward theta only
            W = None
        else:
            W = self.voxel_deformer(xyz_canonical)  # B,N,24+K
        return W, A


########################################


# camera sampling (previously in /lib_guidance/camera_sampling.py)
def sample_camera(
    global_step=1,
    n_view=4,
    real_batch_size=1,
    random_azimuth_range=[-180.0, 180.0],
    random_elevation_range=[0.0, 30.0],
    eval_elevation_deg=15,
    camera_distance_range=[0.8, 1.0],  # relative
    fovy_range=[15, 60],
    zoom_range=[1.0, 1.0],
    progressive_until=0,
    relative_radius=True,
):

    # ! from uncond.py
    # ThreeStudio has progressive increase of camera poses, from eval to random
    r = min(1.0, global_step / (progressive_until + 1))
    elevation_range = [
        (1 - r) * eval_elevation_deg + r * random_elevation_range[0],
        (1 - r) * eval_elevation_deg + r * random_elevation_range[1],
    ]
    azimuth_range = [
        (1 - r) * 0.0 + r * random_azimuth_range[0],
        (1 - r) * 0.0 + r * random_azimuth_range[1],
    ]

    # sample elevation angles
    if random.random() < 0.5:
        # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
        elevation_deg = (
            torch.rand(real_batch_size) * (elevation_range[1] - elevation_range[0])
            + elevation_range[0]
        ).repeat_interleave(n_view, dim=0)
        elevation = elevation_deg * math.pi / 180
    else:
        # otherwise sample uniformly on sphere
        elevation_range_percent = [
            (elevation_range[0] + 90.0) / 180.0,
            (elevation_range[1] + 90.0) / 180.0,
        ]
        # inverse transform sampling
        elevation = torch.asin(
            2
            * (
                torch.rand(real_batch_size)
                * (elevation_range_percent[1] - elevation_range_percent[0])
                + elevation_range_percent[0]
            )
            - 1.0
        ).repeat_interleave(n_view, dim=0)
        elevation_deg = elevation / math.pi * 180.0

    # sample azimuth angles from a uniform distribution bounded by azimuth_range
    # ensures sampled azimuth angles in a batch cover the whole range
    azimuth_deg = (
        torch.rand(real_batch_size).reshape(-1, 1) + torch.arange(n_view).reshape(1, -1)
    ).reshape(-1) / n_view * (azimuth_range[1] - azimuth_range[0]) + azimuth_range[0]
    azimuth = azimuth_deg * math.pi / 180

    ######## Different from original ########
    # sample fovs from a uniform distribution bounded by fov_range
    fovy_deg = (
        torch.rand(real_batch_size) * (fovy_range[1] - fovy_range[0]) + fovy_range[0]
    ).repeat_interleave(n_view, dim=0)
    fovy = fovy_deg * math.pi / 180

    # sample distances from a uniform distribution bounded by distance_range
    camera_distances = (
        torch.rand(real_batch_size)
        * (camera_distance_range[1] - camera_distance_range[0])
        + camera_distance_range[0]
    ).repeat_interleave(n_view, dim=0)
    if relative_radius:
        scale = 1 / torch.tan(0.5 * fovy)
        camera_distances = scale * camera_distances

    # zoom in by decreasing fov after camera distance is fixed
    zoom = (
        torch.rand(real_batch_size) * (zoom_range[1] - zoom_range[0]) + zoom_range[0]
    ).repeat_interleave(n_view, dim=0)
    fovy = fovy * zoom
    fovy_deg = fovy_deg * zoom
    ###########################################

    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    azimuth, elevation
    # build opencv camera
    z = -torch.stack(
        [
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        -1,
    )  # nview, 3
    # up is 0,0,1
    x = torch.cross(
        z, torch.tensor([0.0, 0.0, 1.0], device=z.device).repeat(n_view, 1), -1
    )
    y = torch.cross(z, x, -1)

    R_wc = torch.stack([x, y, z], dim=2)  # nview, 3, 3, col is basis
    t_wc = camera_positions

    T_wc = torch.eye(4, device=R_wc.device).repeat(n_view, 1, 1)
    T_wc[:, :3, :3] = R_wc
    T_wc[:, :3, 3] = t_wc

    return T_wc, fovy_deg  # B,4,4, B


def opencv2blender(T):
    ret = T.clone()
    # y,z are negative
    ret[:, :, 1] *= -1
    ret[:, :, 2] *= -1
    return ret


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


########################################


def create_log(log_dir):
    os.makedirs(osp.join(log_dir, "viz_step"), exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    # configure logging
    logging.getLogger().handlers.clear()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    logger_handler.addFilter(HostnameFilter())
    formatter = logging.Formatter(
        "| %(hostname)s | %(levelname)s | %(asctime)s | %(message)s   [%(filename)s:%(lineno)d]",
        "%b-%d-%H:%M:%S",
    )
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_logger_handler = logging.FileHandler(
        osp.join(log_dir, f"{current_datetime}.log")
    )
    file_logger_handler.setFormatter(formatter)
    logger.addHandler(file_logger_handler)
    return writer


class TGFitter:
    def __init__(
        self,
        log_dir,
        profile_fn="./profiles/zju_3m.yaml",
        device=torch.device("cuda:0"),
        debug: bool = True,
    ):
        """the __init__ method basically just reads the configs and sets up the logging; really not much going on here"""
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.profile_fn = profile_fn
        shutil.copy(profile_fn, osp.join(self.log_dir, osp.basename(profile_fn)))
        self.mode = "human"
        self.device = device

        # assign vars in zju_3m.yaml to self attrs
        for k, v in OmegaConf.load(profile_fn).items():
            setattr(self, k, v)

        # prepare base R
        self.viz_base_R = (
            torch.from_numpy(np.asarray(euler2mat(np.pi, 0, 0, "sxyz")))
            .float()
            .to(self.device)
        )

        self.reg_base_R_global = (
            matrix_to_axis_angle(
                torch.as_tensor(euler2mat(np.pi / 2.0, 0, np.pi / 2.0, "sxyz"))[None]
            )[0]
            .float()
            .to(self.device)
        )

        self.writer: SummaryWriter = create_log(self.log_dir)
        return

    def load_saved_model(self, ckpt_path=None):
        """load a saved model from a checkpoint path"""
        if ckpt_path is None:
            ckpt_path = osp.join(self.log_dir, "model.pth")
        ret = self._get_model_optimizer(betas=None)
        model = ret[0]

        model.load(torch.load(ckpt_path))
        model.to(self.device)
        model.eval()
        logging.info("After loading:")
        model.summary()
        return model

    def _get_model_optimizer(self, betas):
        seed_everything(self.SEED)

        template = SMPLTemplate(
            smpl_model_path="data/smpl_model/SMPL_NEUTRAL.pkl",
            init_beta=betas,
            cano_pose_type="da_pose",
            voxel_deformer_res=64,
        )

        model = GaussianTemplateModel(
            template=template,
            betas=betas,
            w_correction_flag=self.W_CORRECTION_FLAG,
            w_rest_dim=0,  # 4 #0 #16
            f_localcode_dim=0,
            max_sph_order=1,  # 0
            w_memory_type="voxel",  # "point"
            max_scale=1.0,  # 0.03
            min_scale=0.0,  # 0.0003 #0.003 #3
            # * init
            opacity_init_value=0.99,  # 0.9
            # on mesh init
            onmesh_init_subdivide_num=0,
            onmesh_init_scale_factor=1.0,
            onmesh_init_thickness_factor=0.5,
        ).to(self.device)

        logging.info(f"Init with {model.N} Gaussians")

        # * set optimizer
        LR_SPH_REST = self.LR_SPH / 20.0
        optimizer = torch.optim.Adam(
            model.get_optimizable_list(
                lr_p=self.LR_P,
                lr_o=self.LR_O,
                lr_s=self.LR_S,
                lr_q=self.LR_Q,
                lr_sph=self.LR_SPH,
                lr_sph_rest=LR_SPH_REST,
                lr_w=self.LR_W,
                lr_w_rest=self.LR_W_REST,
                lr_f=0.0,
            ),
        )

        xyz_scheduler_func = get_expon_lr_func(
            lr_init=self.LR_P,
            lr_final=self.LR_P_FINAL,
            lr_delay_mult=0.01,  # 0.02
            max_steps=self.TOTAL_steps,
        )

        init_step = 1500  # 1000 #500 #2000 #300 #2000
        w_dc_scheduler_func = get_expon_lr_func_interval(
            init_step=init_step,
            final_step=self.TOTAL_steps,
            lr_init=self.LR_W,
            lr_final=0.00001,
            lr_delay_mult=0.01,  # 0.02
        )
        w_rest_scheduler_func = get_expon_lr_func_interval(
            init_step=init_step,
            final_step=self.TOTAL_steps,
            lr_init=0.00003,
            lr_final=0.000003,
            lr_delay_mult=0.01,
        )
        sph_scheduler_func = get_expon_lr_func(
            lr_init=self.LR_SPH,
            lr_final=self.LR_SPH,
            lr_delay_mult=0.01,  # 0.02
            max_steps=self.TOTAL_steps,
        )
        sph_rest_scheduler_func = get_expon_lr_func(
            lr_init=LR_SPH_REST,
            lr_final=LR_SPH_REST,
            lr_delay_mult=0.01,  # 0.02
            max_steps=self.TOTAL_steps,
        )

        return (
            model,
            optimizer,
            xyz_scheduler_func,
            w_dc_scheduler_func,
            w_rest_scheduler_func,
            sph_scheduler_func,
            sph_rest_scheduler_func,
        )

    def _get_pose_optimizer(self, data_provider):

        # * prepare pose optimizer list and the schedulers
        scheduler_dict, pose_optim_l = {}, []
        if data_provider is not None:
            start_step = 1500  # 500 #1000
            end_step = self.TOTAL_steps
            pose_optim_l.extend(
                [
                    {
                        "params": data_provider.pose_base_list,
                        "lr": self.POSE_R_BASE_LR,
                        "name": "pose_base",
                    },
                    {
                        "params": data_provider.pose_rest_list,
                        "lr": self.POSE_R_REST_LR,
                        "name": "pose_rest",
                    },
                    {
                        "params": data_provider.global_trans_list,
                        "lr": self.POSE_T_LR,
                        "name": "pose_trans",
                    },
                ]
            )
            scheduler_dict["pose_base"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_R_BASE_LR,
                lr_final=0.0001,
                lr_delay_mult=0.01,  # 0.02
            )
            scheduler_dict["pose_rest"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_R_REST_LR,
                lr_final=0.0001,
                lr_delay_mult=0.01,  # 0.02
            )
            scheduler_dict["pose_trans"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_T_LR,
                lr_final=0.0001,
                lr_delay_mult=0.01,  # 0.02
            )

        optimizer_smpl = torch.optim.Adam(pose_optim_l)
        return optimizer_smpl, scheduler_dict

    def _fit_step(
        self,
        model,
        data_pack,
        act_sph_ord,
        random_bg=True,
        scale_multiplier=1.0,
        opacity_multiplier=1.0,
        opa_th=-1,
        use_box_crop_pad=-1,
        default_bg=[1.0, 1.0, 1.0],
    ):
        gt_rgb, gt_mask, K, pose_base, pose_rest, global_trans, time_index = data_pack
        gt_rgb = gt_rgb.clone()

        pose = torch.cat([pose_base, pose_rest], dim=1)
        H, W = gt_rgb.shape[1:3]
        additional_dict = {"t": time_index}

        mu, fr, sc, op, sph, additional_ret = model(
            pose,
            global_trans,
            additional_dict=additional_dict,
            active_sph_order=act_sph_ord,
        )

        sc = sc * scale_multiplier
        op = op * opacity_multiplier

        loss_recon = 0.0
        loss_lpips, loss_ssim = (
            torch.zeros(1).to(gt_rgb.device).squeeze(),
            torch.zeros(1).to(gt_rgb.device).squeeze(),
        )
        loss_mask = 0.0
        render_pkg_list, rgb_target_list, mask_target_list = [], [], []
        for i in range(len(gt_rgb)):
            if random_bg:
                bg = np.random.uniform(0.0, 1.0, size=3)
            else:
                bg = np.array(default_bg)
            bg_tensor = torch.from_numpy(bg).float().to(gt_rgb.device)
            gt_rgb[i][gt_mask[i] == 0] = bg_tensor[None, None, :]
            render_pkg = render_cam_pcl(
                mu[i], fr[i], sc[i], op[i], sph[i], H, W, K[i], False, act_sph_ord, bg
            )
            if opa_th > 0.0:
                bg_mask = render_pkg["alpha"][0] < opa_th
                render_pkg["rgb"][:, bg_mask] = (
                    render_pkg["rgb"][:, bg_mask] * 0.0 + default_bg[0]
                )

            if use_box_crop_pad > 0:
                # pad the gt mask and crop the image, for a large image with a small fg object
                _yl, _yr, _xl, _xr = get_bbox(gt_mask[i], use_box_crop_pad)
                for key in ["rgb", "alpha", "dep"]:
                    render_pkg[key] = render_pkg[key][:, _yl:_yr, _xl:_xr]
                rgb_target = gt_rgb[i][_yl:_yr, _xl:_xr]
                mask_target = gt_mask[i][_yl:_yr, _xl:_xr]
            else:
                rgb_target = gt_rgb[i]
                mask_target = gt_mask[i]

            # * standard recon loss
            _loss_recon = abs(render_pkg["rgb"].permute(1, 2, 0) - rgb_target).mean()
            _loss_mask = abs(render_pkg["alpha"].squeeze(0) - mask_target).mean()

            # * ssim
            if self.LAMBDA_SSIM > 0:
                _loss_ssim = 1.0 - ssim(
                    render_pkg["rgb"][None],
                    rgb_target.permute(2, 0, 1)[None],
                )
                loss_ssim = loss_ssim + _loss_ssim

            loss_recon = loss_recon + _loss_recon
            loss_mask = loss_mask + _loss_mask
            render_pkg_list.append(render_pkg)
            rgb_target_list.append(rgb_target)
            mask_target_list.append(mask_target)

        loss_recon = loss_recon / len(gt_rgb)
        loss_mask = loss_mask / len(gt_rgb)
        loss_lpips = loss_lpips / len(gt_rgb)
        loss_ssim = loss_ssim / len(gt_rgb)

        loss = (
            loss_recon + self.LAMBDA_SSIM * loss_ssim + self.LAMBDA_LPIPS * loss_lpips
        )

        return (
            loss,
            loss_mask,
            render_pkg_list,
            rgb_target_list,
            mask_target_list,
            (mu, fr, sc, op, sph),
            {
                "loss_l1_recon": loss_recon,
                "loss_lpips": loss_lpips,
                "loss_ssim": loss_ssim,
            },
        )

    def _compute_reg3D(self, model):
        K = 6
        (
            q_std,
            s_std,
            o_std,
            cd_std,
            ch_std,
            w_std,
            w_rest_std,
            f_std,
            w_norm,
            w_rest_norm,
            dist_sq,
            max_s_sq,
        ) = model.compute_reg(K)

        lambda_std_q = 0.01
        lambda_std_s = 0.01
        lambda_std_o = 0.01
        lambda_std_cd = 0.01
        lambda_std_ch = 0.01
        lambda_std_w = 0.3
        lambda_std_w_rest = 0.5
        lambda_small_scale = 0.01
        lambda_w_norm = 0.01
        lambda_w_rest_norm = 0.01
        lambda_std_f = lambda_std_w
        lambda_knn_dist = 0.0

        # regression loss L_reg
        reg_loss = (
            lambda_std_q * q_std
            + lambda_std_s * s_std
            + lambda_std_o * o_std
            + lambda_std_cd * cd_std
            + lambda_std_ch * ch_std
            + lambda_knn_dist * dist_sq  # 0.0
            + lambda_std_w * w_std
            + lambda_std_w_rest * w_rest_std
            + lambda_std_f * f_std
            + lambda_w_norm * w_norm
            + lambda_w_rest_norm * w_rest_norm
            + lambda_small_scale * max_s_sq
        )

        details = {
            "q_std": q_std.detach(),
            "s_std": s_std.detach(),
            "o_std": o_std.detach(),
            "cd_std": cd_std.detach(),
            "ch_std": ch_std.detach(),
            "w_std": w_std.detach(),
            "w_rest_std": w_rest_std.detach(),
            "f_std": f_std.detach(),
            "knn_dist_sq": dist_sq.detach(),
            "w_norm": w_norm.detach(),
            "w_rest_norm": w_rest_norm.detach(),
            "max_s_sq": max_s_sq.detach(),
        }
        return reg_loss, details

    def _add_scalar(self, *args, **kwargs):
        if self.FAST_TRAINING:
            return
        self.writer.add_scalar(*args, **kwargs)
        return

    def testtime_pose_optimization(
        self,
        data_pack,
        model,
        evaluator,
        pose_base_lr=3e-3,
        pose_rest_lr=3e-3,
        trans_lr=3e-3,
        steps=100,
        decay_steps=30,
        decay_factor=0.5,
        check_every_n_step=5,
        viz_fn=None,
    ):
        # * Like Instant avatar, optimize the smpl pose f
        # * will optimize all poses in the data_pack
        torch.cuda.empty_cache()
        seed_everything(self.SEED)
        model.eval()  # to get gradients, but never optimized
        evaluator.eval()
        gt_rgb, gt_mask, K, pose_b, pose_r, trans = data_pack[:6]
        pose_b = pose_b.detach().clone()
        pose_r = pose_r.detach().clone()
        trans = trans.detach().clone()
        pose_b.requires_grad_(True)
        pose_r.requires_grad_(True)
        trans.requires_grad_(True)
        gt_rgb, gt_mask = gt_rgb.to(self.device), gt_mask.to(self.device)

        optim_l = [
            {"params": [pose_b], "lr": pose_base_lr},
            {"params": [pose_r], "lr": pose_rest_lr},
            {"params": [trans], "lr": trans_lr},
        ]
        optimizer_smpl = torch.optim.SGD(optim_l)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_smpl, step_size=decay_steps, gamma=decay_factor
        )

        loss_list, psnr_list, ssim_list, lpips_list = [], [], [], []
        viz_step_list = []
        for inner_step in range(steps):
            # optimize
            optimizer_smpl.zero_grad()
            loss_recon, _, rendered_list, _, _, _, _ = self._fit_step(
                model,
                [gt_rgb, gt_mask, K, pose_b, pose_r, trans, None],
                act_sph_ord=model.max_sph_order,
                random_bg=False,
                default_bg=[0.0, 0.0, 0.0],
            )
            loss = loss_recon
            loss.backward()
            optimizer_smpl.step()
            scheduler.step()
            loss_list.append(float(loss))

            if (
                inner_step % check_every_n_step == 0
                or inner_step == steps - 1
                and viz_fn is not None
            ):
                viz_step_list.append(inner_step)
                with torch.no_grad():
                    _check_results = [
                        evaluator(
                            rendered_list[0]["rgb"].permute(1, 2, 0)[None],
                            gt_rgb[0][None],
                        )
                    ]
                psnr = torch.stack([r["psnr"] for r in _check_results]).mean().item()
                ssim = torch.stack([r["ssim"] for r in _check_results]).mean().item()
                lpips = torch.stack([r["lpips"] for r in _check_results]).mean().item()
                psnr_list.append(float(psnr))
                ssim_list.append(float(ssim))
                lpips_list.append(float(lpips))

        if viz_fn is not None:
            os.makedirs(osp.dirname(viz_fn), exist_ok=True)
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 4, 1)
            plt.plot(viz_step_list, psnr_list)
            plt.title(f"PSNR={psnr_list[-1]}"), plt.grid()
            plt.subplot(1, 4, 2)
            plt.plot(viz_step_list, ssim_list)
            plt.title(f"SSIM={ssim_list[-1]}"), plt.grid()
            plt.subplot(1, 4, 3)
            plt.plot(viz_step_list, lpips_list)
            plt.title(f"LPIPS={lpips_list[-1]}"), plt.grid()
            plt.subplot(1, 4, 4)
            plt.plot(loss_list)
            plt.title(f"Loss={loss_list[-1]}"), plt.grid()
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(viz_fn)
            plt.close()

        return (
            pose_b.detach().clone(),
            pose_r.detach().clone(),
            trans.detach().clone(),
        )

    @torch.no_grad()
    def eval_fps(self, model, real_data_provider, rounds=1):
        model.eval()
        model.cache_for_fast()
        logging.info(f"Model has {model.N} points.")
        N_frames = len(real_data_provider.rgb_list)
        ret = real_data_provider(N_frames)
        gt_rgb, gt_mask, K, pose_base, pose_rest, global_trans, time_index = ret
        pose = torch.cat([pose_base, pose_rest], 1)
        H, W = gt_rgb.shape[1:3]
        sph_o = model.max_sph_order
        logging.info(f"FPS eval using active_sph_order: {sph_o}")
        # run one iteration to check the output correctness
        mu, fr, sc, op, sph, additional_ret = model(
            pose[0:1],
            global_trans[0:1],
            additional_dict={},
            active_sph_order=sph_o,
            fast=True,
        )
        bg = [1.0, 1.0, 1.0]
        render_pkg = render_cam_pcl(
            mu[0], fr[0], sc[0], op[0], sph[0], H, W, K[0], False, sph_o, bg
        )
        pred = render_pkg["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        imageio.imsave(osp.join(self.log_dir, "fps_eval_sample.png"), pred)

        logging.info("Start FPS test...")
        start_t = time.time()

        for j in tqdm(range(int(N_frames * rounds))):
            i = j % N_frames
            mu, fr, sc, op, sph, additional_ret = model(
                pose[i : i + 1],
                global_trans[i : i + 1],
                additional_dict={"t": time_index},
                active_sph_order=sph_o,
                fast=True,
            )
            bg = [1.0, 1.0, 1.0]
            render_pkg = render_cam_pcl(
                mu[0], fr[0], sc[0], op[0], sph[0], H, W, K[0], False, sph_o, bg
            )
        end_t = time.time()

        fps = (rounds * N_frames) / (end_t - start_t)
        logging.info(f"FPS: {fps}")
        with open(osp.join(self.log_dir, "fps.txt"), "w") as f:
            f.write(f"FPS: {fps}")
        return fps

    def run(
        self,
        data_provider=None,
    ):
        torch.cuda.empty_cache()

        init_beta = data_provider.betas

        (
            model,
            optimizer,
            xyz_scheduler_func,
            w_dc_scheduler_func,
            w_rest_scheduler_func,
            sph_scheduler_func,
            sph_rest_scheduler_func,
        ) = self._get_model_optimizer(betas=init_beta)

        optimizer_pose, scheduler_pose = self._get_pose_optimizer(data_provider)

        # * Optimization Loop
        stat_n_list = []
        active_sph_order, last_reset_step = 0, -1
        seed_everything(self.SEED)
        running_start_t = time.time()
        logging.info(f"Start training at {running_start_t}")
        for step in tqdm(range(self.TOTAL_steps)):
            update_learning_rate(xyz_scheduler_func(step), "xyz", optimizer)
            update_learning_rate(sph_scheduler_func(step), "f_dc", optimizer)
            update_learning_rate(sph_rest_scheduler_func(step), "f_rest", optimizer)

            update_learning_rate(
                w_dc_scheduler_func(step), ["w_dc", "w_dc_vox"], optimizer
            )
            update_learning_rate(
                w_rest_scheduler_func(step), ["w_rest", "w_rest_vox"], optimizer
            )
            for k, v in scheduler_pose.items():
                update_learning_rate(v(step), k, optimizer_pose)

            if step in self.INCREASE_SPH_STEP:
                active_sph_order += 1
                logging.info(f"active_sph_order: {active_sph_order}")

            # * Recon fitting step
            model.train()
            optimizer.zero_grad(), optimizer_pose.zero_grad()

            loss = 0.0

            real_data_pack = data_provider(self.N_POSES_PER_STEP, continuous=False)

            (
                loss_recon,
                loss_mask,
                render_list,
                gt_rgb,
                gt_mask,
                model_ret,
                loss_dict,
            ) = self._fit_step(
                model,
                data_pack=real_data_pack,
                act_sph_ord=active_sph_order,
                random_bg=True,
                use_box_crop_pad=20,
                default_bg=[0.0, 0.0, 0.0],
            )
            loss = loss + loss_recon
            for k, v in loss_dict.items():
                self._add_scalar(k, v.detach(), step)

            if (
                last_reset_step < 0
                or step - last_reset_step > self.MASK_LOSS_PAUSE_AFTER_RESET
                and step > 0
            ):
                loss = loss + self.LAMBDA_MASK * loss_mask
                self._add_scalar("loss_mask", loss_mask.detach(), step)
            self._add_scalar("loss", loss.detach(), step)

            # * Reg Terms
            reg_loss, reg_details = self._compute_reg3D(model)
            loss = reg_loss + loss
            for k, v in reg_details.items():
                self._add_scalar(k, v.detach(), step)

            loss.backward()
            optimizer.step()

            if step > 1500:
                optimizer_pose.step()

            self._add_scalar("N", model.N, step)

            # * Gaussian Control
            if step > self.DENSIFY_START:
                for render_pkg in render_list:
                    model.record_xyz_grad_radii(
                        render_pkg["viewspace_points"],
                        render_pkg["radii"],
                        render_pkg["visibility_filter"],
                    )

            if (
                step > self.DENSIFY_START
                and step < 2500  # 10000 #15000
                and step % self.DENSIFY_INTERVAL == 0
            ):
                N_old = model.N
                model.densify(
                    optimizer=optimizer,
                    max_grad=self.MAX_GRAD,
                    percent_dense=self.PERCENT_DENSE,
                    extent=0.5,
                    verbose=True,
                )
                logging.info(f"Densify: {N_old}->{model.N}")

            if step > self.PRUNE_START and step % self.PRUNE_INTERVAL == 0:
                N_old = model.N
                model.prune_points(
                    optimizer,
                    min_opacity=self.OPACIT_PRUNE_TH,
                    max_screen_size=1e10,  # ! disabled
                )
                logging.info(f"Prune: {N_old}->{model.N}")

            if (step + 1) in self.RESET_OPACITY_STEPS:
                model.reset_opacity(optimizer, self.OPACIT_RESET_VALUE)
                last_reset_step = step

            if (step + 1) in self.REGAUSSIAN_STEPS:
                model.regaussian(optimizer, self.REGAUSSIAN_STD)

            stat_n_list.append(model.N)

            if self.FAST_TRAINING:
                continue

            # * Viz
            if (step + 1) % 500 == 0 or step == 0:
                mu, fr, s, o, sph = model_ret[:5]
                save_path = f"{self.log_dir}/viz_step/step_{step}.png"
                viz_render(gt_rgb[0], gt_mask[0], render_list[0], save_path=save_path)
                # viz the spinning in the middle
                (
                    _,
                    _,
                    K,
                    pose_base,
                    pose_rest,
                    global_trans,
                    time_index,
                ) = real_data_pack
                viz_spinning(
                    model,
                    torch.cat([pose_base, pose_rest], 1)[:1],
                    global_trans[:1],
                    data_provider.H,
                    data_provider.W,
                    K[0],
                    save_path=f"{self.log_dir}/viz_step/spinning_{step}.gif",
                    time_index=time_index,
                    active_sph_order=active_sph_order,
                    bg_color=[0.0, 0.0, 0.0],
                )

                can_pose = model.template.canonical_pose.detach()
                can_pose[0] = self.viz_base_R.to(can_pose.device)
                can_pose = matrix_to_axis_angle(can_pose)[None]
                can_trans = torch.zeros(len(can_pose), 3).to(can_pose)
                can_trans[:, -1] = 3.0
                viz_H, viz_W = 512, 512
                viz_K = fov2K(60, viz_H, viz_W)
                viz_spinning(
                    model,
                    can_pose,
                    can_trans,
                    viz_H,
                    viz_W,
                    viz_K,
                    save_path=f"{self.log_dir}/viz_step/spinning_can_{step}.gif",
                    time_index=None,  # canonical pose use t=None
                    active_sph_order=active_sph_order,
                    bg_color=[0.0, 0.0, 0.0],
                )

                # viz the distrbution
                plt.figure(figsize=(15, 3))
                scale = model.get_s
                for i in range(3):
                    _s = scale[:, i].detach().cpu().numpy()
                    plt.subplot(1, 3, i + 1)
                    plt.hist(_s, bins=100)
                    plt.title(f"scale {i}")
                    plt.grid(), plt.ylabel("count"), plt.xlabel("scale")
                plt.tight_layout()
                plt.savefig(f"{self.log_dir}/viz_step/s_hist_step_{step}.png")
                plt.close()

                plt.figure(figsize=(12, 4))
                opacity = (model.get_o).squeeze(-1)
                plt.hist(opacity.detach().cpu().numpy(), bins=100)
                plt.title(f"opacity")
                plt.grid(), plt.ylabel("count"), plt.xlabel("opacity")
                plt.tight_layout()
                plt.savefig(f"{self.log_dir}/viz_step/o_hist_step_{step}.png")
                plt.close()

        running_end_t = time.time()
        logging.info(
            f"Training time: {(running_end_t - running_start_t):.3f} seconds i.e. {(running_end_t - running_start_t)/60.0 :.3f} minutes"
        )

        # * save
        model.eval()
        ckpt_path = f"{self.log_dir}/model.pth"
        logging.info(f"Saving model to {ckpt_path}...")
        torch.save(model.state_dict(), ckpt_path)
        pose_path = f"{self.log_dir}/training_poses.pth"
        torch.save(data_provider.state_dict(), pose_path)
        model.to("cpu")

        # * stat
        plt.figure(figsize=(5, 5))
        plt.plot(stat_n_list)
        plt.title("N"), plt.xlabel("step"), plt.ylabel("N")
        plt.savefig(f"{self.log_dir}/N.png")
        plt.close()

        model.summary()

        # move the model back to the device
        model.to(self.device)
        return model, data_provider
