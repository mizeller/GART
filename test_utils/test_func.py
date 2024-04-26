import sys, os, os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(osp.dirname(osp.abspath(__file__)))

import torch
import numpy as np
from eval_utils_instant_nvr import Evaluator as EvalNVR

from lib_render.gauspl_renderer import render_cam_pcl
import cv2, glob
import pandas as pd
from tqdm import tqdm
from lib_data.zju_mocap import Dataset as ZJUDataset, get_batch_sampler
import logging


def _get_evaluator(mode, device):
    evaluator = EvalNVR()
    evaluator = evaluator.to(device)
    evaluator.eval()
    return evaluator


def test(
    solver,
    seq_name: str,
    tto_flag=True,
    tto_step=300,
    tto_decay=60,
    tto_decay_factor=0.5,
    pose_base_lr=3e-3,
    pose_rest_lr=3e-3,
    trans_lr=3e-3,
    dataset_mode="zju",
):
    device = solver.device
    model = solver.load_saved_model()
    eval_mode = "nvr"
    test_dataset = ZJUDataset(
        data_root="./data/zju_mocap",
        video_name=seq_name,
        split="test",
        image_zoom_ratio=0.5,
    )
    bg = [0.0, 0.0, 0.0]  # zju use black background

    evaluator = _get_evaluator(eval_mode, device)

    _save_eval_maps(
        solver.log_dir,
        "test",
        model,
        solver,
        test_dataset,
        dataset_mode=dataset_mode,
        device=device,
        bg=bg,
        tto_flag=tto_flag,
        tto_step=tto_step,
        tto_decay=tto_decay,
        tto_decay_factor=tto_decay_factor,
        tto_evaluator=evaluator,
        pose_base_lr=pose_base_lr,
        pose_rest_lr=pose_rest_lr,
        trans_lr=trans_lr,
    )

    if tto_flag:
        _evaluate_dir(evaluator, solver.log_dir, "test_tto")
    else:
        _evaluate_dir(evaluator, solver.log_dir, "test")

    return


def _save_eval_maps(
    log_dir,
    save_name,
    model,
    solver,
    test_dataset,
    dataset_mode="people_snapshot",
    bg=[1.0, 1.0, 1.0],
    # tto
    tto_flag=False,
    tto_step=300,
    tto_decay=60,
    tto_decay_factor=0.5,
    tto_evaluator=None,
    pose_base_lr=3e-3,
    pose_rest_lr=3e-3,
    trans_lr=3e-3,
    device=torch.device("cuda:0"),
):
    model.eval()
    if tto_flag:
        test_save_dir_tto = osp.join(log_dir, f"{save_name}_tto")
        os.makedirs(test_save_dir_tto, exist_ok=True)
    else:
        test_save_dir = osp.join(log_dir, save_name)
        os.makedirs(test_save_dir, exist_ok=True)

    # ! follow instant-nvr evaluation
    iter_test_dataset = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=get_batch_sampler(test_dataset, frame_sampler_interval=6),
        num_workers=0,
    )

    logging.info(f"Saving images [TTO={tto_flag}] [N={len(iter_test_dataset)}]...")
    for batch_idx, batch in tqdm(enumerate(iter_test_dataset)):
        # get data
        data, meta = batch

        for k in data.keys():
            data[k] = data[k].squeeze(0)

        rgb_gt = torch.as_tensor(data["rgb"])[None].float().to(device)
        mask_gt = torch.as_tensor(data["mask"])[None].float().to(device)
        H, W = rgb_gt.shape[1:3]
        K = torch.as_tensor(data["K"]).float().to(device)
        pose = torch.as_tensor(data["smpl_pose"]).float().to(device)[None]
        trans = torch.as_tensor(data["smpl_trans"]).float().to(device)[None]
        fn = f"frame{int(meta['frame_idx']):04d}_view{int(meta['cam_ind']):04d}.png"
        if tto_flag:
            # change the pose from the dataset to fit the test view
            pose_b, pose_r = pose[:, :1], pose[:, 1:]
            model.eval()
            new_pose_b, new_pose_r, new_trans = solver.testtime_pose_optimization(
                data_pack=[
                    rgb_gt,
                    mask_gt,
                    K[None],
                    pose_b,
                    pose_r,
                    trans,
                    None,
                ],
                model=model,
                evaluator=tto_evaluator,
                pose_base_lr=pose_base_lr,
                pose_rest_lr=pose_rest_lr,
                trans_lr=trans_lr,
                steps=tto_step,
                decay_steps=tto_decay,
                decay_factor=tto_decay_factor,
            )
            pose = torch.cat([new_pose_b, new_pose_r], dim=1).detach()
            trans = new_trans.detach()

            save_fn = osp.join(test_save_dir_tto, fn)
            _save_render_image_from_pose(
                model,
                pose,
                trans,
                H,
                W,
                K,
                bg,
                rgb_gt,
                save_fn,
                time_index=batch_idx,
            )
        else:
            save_fn = osp.join(test_save_dir, fn)
            _save_render_image_from_pose(
                model, pose, trans, H, W, K, bg, rgb_gt, save_fn, time_index=batch_idx
            )
    return


@torch.no_grad()
def _save_render_image_from_pose(
    model, pose, trans, H, W, K, bg, rgb_gt, save_fn, time_index=None
):
    act_sph_order = model.max_sph_order
    device = pose.device
    # TODO: handle novel time!, not does not pass in means t=None; Can always use TTO to directly find As!
    additional_dict = {"t": time_index}
    mu, fr, sc, op, sph, _ = model(
        pose, trans, additional_dict=additional_dict, active_sph_order=act_sph_order
    )  # TODO: directly input optimized As!
    render_pkg = render_cam_pcl(
        mu[0], fr[0], sc[0], op[0], sph[0], H, W, K, False, act_sph_order, bg
    )
    mask = (render_pkg["alpha"].squeeze(0) > 0.0).bool()
    render_pkg["rgb"][:, ~mask] = bg[0]  # either 0.0 or 1.0
    pred_rgb = render_pkg["rgb"]  # 3,H,W
    pred_rgb = pred_rgb.permute(1, 2, 0)[None]  # 1,H,W,3

    errmap = (pred_rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy()[0] / np.sqrt(3)
    errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    errmap = torch.from_numpy(errmap).to(device)[None] / 255
    img = torch.cat(
        [rgb_gt[..., [2, 1, 0]], pred_rgb[..., [2, 1, 0]], errmap], dim=2
    )  # ! note, here already swapped the channel order
    cv2.imwrite(save_fn, img.cpu().numpy()[0] * 255)
    return


@torch.no_grad()
def _evaluate_dir(evaluator, log_dir, dir_name="test", device=torch.device("cuda:0")):
    imgs = [cv2.imread(fn) for fn in glob.glob(f"{osp.join(log_dir, dir_name)}/*.png")]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = [torch.tensor(img).float() / 255.0 for img in imgs]
    evaluator = evaluator.to(device)
    evaluator.eval()
    eval_mode = "instant-nvr"
    H, W = imgs[0].shape[:2]
    logging.info(f"Image size: {H}x{W}")
    W //= 3
    with torch.no_grad():
        results = [
            evaluator(
                img[None, :, W : 2 * W].to(device),
                img[None, :, :W].to(device),
            )
            for img in imgs
        ]

    ret = {}
    logging.info(f"Eval with {eval_mode} Evaluator from their original code release")
    with open(
        osp.join(log_dir, f"results_{dir_name}_evaluator_{eval_mode}.txt"), "w"
    ) as f:
        psnr = torch.stack([r["psnr"] for r in results]).mean().item()
        logging.info(f"[{dir_name}] PSNR: {psnr:.2f}")
        f.write(f"[{dir_name}] PSNR: {psnr:.2f}\n")
        ret["psnr"] = psnr

        ssim = torch.stack([r["ssim"] for r in results]).mean().item()
        logging.info(f"[{dir_name}] SSIM: {ssim:.4f}")
        f.write(f"[{dir_name}] SSIM: {ssim:.4f}\n")
        ret["ssim"] = ssim

        lpips = torch.stack([r["lpips"] for r in results]).mean().item()
        logging.info(f"[{dir_name}] LPIPS: {lpips:.4f}")
        f.write(f"[{dir_name}] LPIPS: {lpips:.4f}\n")
        ret["lpips"] = lpips
    # save a xls of the per frame results
    for i in range(len(results)):
        for k in results[i].keys():
            results[i][k] = float(results[i][k].cpu())
    df = pd.DataFrame(results)
    df.to_excel(osp.join(log_dir, f"results_{dir_name}_evaluator_{eval_mode}.xlsx"))

    metrics = {
        "psnr": [r["psnr"] for r in results],
        "ssim": [r["ssim"] for r in results],
        "lpips": [r["lpips"] for r in results],
    }
    np.save(osp.join(log_dir, f"{dir_name}_{eval_mode}.npy"), metrics)
    return ret
