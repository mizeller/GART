# A template only handle the query of the
import sys, os.path as osp

sys.path.append(osp.dirname(osp.abspath(__file__)))

import numpy as np
import torch
from torch import nn

from smplx.smplx import SMPLLayer
from smplx.smplx.lbs import batch_rigid_transform
from voxel_deformer import VoxelDeformer

from pytorch3d.transforms import axis_angle_to_matrix
from model_utils import get_predefined_human_rest_pose


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
