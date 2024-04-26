from transforms3d.axangles import mat2axangle, axangle2mat
from pytorch3d.transforms import axis_angle_to_matrix

from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import imageio
import cv2
import torch
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))

from smplx.smplx import SMPLLayer


class Dataset(Dataset):
    # from instant avatar
    def __init__(
        self,
        data_root="./data/zju_mocap",
        video_name="my_392",
        split="train",
        image_zoom_ratio=0.5,
        bg_color=0.0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.video_name = video_name
        self.image_zoom_ratio = image_zoom_ratio
        self.bg_color = bg_color

        root = osp.join(data_root, video_name)  # data/zju-mocap/my_392
        anno_fn = osp.join(root, "annots.npy")  # data/zju-mocap/my_392/annots.npy
        annots = np.load(anno_fn, allow_pickle=True).item()  # keys: ims, cams,
        self.cams = annots["cams"]

        # ! Check the run.py in instant-nvr evaluation
        num_cams = len(self.cams["K"])  # ∃ 23 cams
        training_view = [4]
        test_view = [
            i for i in range(num_cams) if i not in training_view
        ]  # all views apart from training_view (4)

        if split == "train" or split == "prune":
            self.view = training_view
        else:  # split == "test":
            self.view = test_view

        self.ims = np.array(
            [
                np.array(ims_data["ims"])[self.view]
                for ims_data in annots["ims"][0 : 100 * 5][::5]
            ]
        ).ravel()
        self.cam_inds = np.array(
            [
                np.arange(len(ims_data["ims"]))[self.view]
                for ims_data in annots["ims"][0 : 100 * 5][::5]
            ]
        ).ravel()
        self.num_cams = len(self.view)

        # ! the cams R is in a very low precision, have use SVD to project back to SO(3)
        for cid in range(num_cams):
            _R = self.cams["R"][cid]
            u, s, vh = np.linalg.svd(_R)
            new_R = u @ vh
            self.cams["R"][cid] = new_R

        # this is copied
        smpl_layer = SMPLLayer(
            osp.join(osp.dirname(__file__), "../data/smpl-meta/SMPL_NEUTRAL.pkl")
        )

        # * Load smpl to camera frame
        self.smpl_theta_list, self.smpl_trans_list, smpl_beta_list = [], [], []
        self.meta = []
        for img_fn in self.ims:
            cam_ind = int(img_fn.split("/")[-2])
            frame_idx = int(img_fn.split("/")[-1].split(".")[0])
            self.meta.append({"cam_ind": cam_ind, "frame_idx": frame_idx})
            smpl_fn = osp.join(root, "smpl_params", f"{frame_idx}.npy")
            smpl_data = np.load(smpl_fn, allow_pickle=True).item()
            T_cw = np.eye(4)
            T_cw[:3, :3], T_cw[:3, 3] = (
                np.array(self.cams["R"][cam_ind]),
                np.array(self.cams["T"][cam_ind]).squeeze(-1) / 1000.0,
            )

            smpl_theta = smpl_data["poses"].reshape((24, 3))
            assert np.allclose(smpl_theta[0], 0)
            smpl_rot, smpl_trans = smpl_data["Rh"][0], smpl_data["Th"]
            smpl_R = axangle2mat(
                smpl_rot / (np.linalg.norm(smpl_rot) + 1e-6), np.linalg.norm(smpl_rot)
            )

            T_wh = np.eye(4)
            T_wh[:3, :3], T_wh[:3, 3] = smpl_R.copy(), smpl_trans.squeeze(0).copy()

            T_ch = T_cw.astype(np.float64) @ T_wh.astype(np.float64)

            smpl_global_rot_d, smpl_global_rot_a = mat2axangle(T_ch[:3, :3])
            smpl_global_rot = smpl_global_rot_d * smpl_global_rot_a
            smpl_trans = T_ch[:3, 3]  # 3
            smpl_theta[0] = smpl_global_rot
            beta = smpl_data["shapes"][0][:10]

            # ! Because SMPL global rot is rot around joint-0, have to correct this in the global translation!!
            _pose = axis_angle_to_matrix(torch.from_numpy(smpl_theta)[None])
            so = smpl_layer(
                torch.from_numpy(beta)[None],
                body_pose=_pose[:, 1:],
            )
            j0 = (so.joints[0, 0]).numpy()
            t_correction = (_pose[0, 0].numpy() - np.eye(3)) @ j0
            smpl_trans = smpl_trans + t_correction

            self.smpl_theta_list.append(smpl_theta)
            smpl_beta_list.append(beta)
            self.smpl_trans_list.append(smpl_trans)

        self.beta = np.array(smpl_beta_list).mean(0)

        return

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, index):
        img_path = osp.join(self.data_root, self.video_name, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        mask_path = osp.join(
            self.data_root,
            self.video_name,
            self.ims[index].replace("images", "mask").replace(".jpg", ".png"),
        )
        msk = imageio.imread(mask_path)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        cam_ind = self.cam_inds[index]
        K = np.array(self.cams["K"][cam_ind])
        D = np.array(self.cams["D"][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        H, W = int(img.shape[0] * self.image_zoom_ratio), int(
            img.shape[1] * self.image_zoom_ratio
        )
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        K[:2] = K[:2] * self.image_zoom_ratio

        img[msk == 0] = self.bg_color

        ret = {
            "rgb": img.astype(np.float32),
            "mask": msk.astype(np.bool).astype(np.float32),
            "K": K.copy().astype(np.float32),
            "smpl_beta": self.beta.astype(np.float32),
            "smpl_pose": self.smpl_theta_list[index].astype(np.float32),
            "smpl_trans": self.smpl_trans_list[index].astype(np.float32),
            "idx": index,
        }

        assert cam_ind == self.meta[index]["cam_ind"]

        meta_info = {
            "video": self.video_name,
            "cam_ind": cam_ind,
            "frame_idx": self.meta[index]["frame_idx"],
        }
        viz_id = f"video{self.video_name}_dataidx{index}"
        meta_info["viz_id"] = viz_id
        return ret, meta_info
