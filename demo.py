"""small debug script to understand the code"""

# default
import torch
import uuid
import logging
import os.path as osp

# custom
from viz_utils import viz_human_all
from test_utils import test
from solver import TGFitter


from lib_data.zju_mocap import Dataset as ZJUDataset
from lib_data.data_provider import RealDataOptimizablePoseProviderPose


def prepare_real_seq(seq_name: str, log_dir: str, device=torch.device("cuda:0")):
    logging.info("Prepare real seq: {}".format(seq_name))
    dataset = ZJUDataset(
        data_root="./data/zju_mocap",
        video_name=seq_name,
        split="train",
        image_zoom_ratio=0.5,
    )

    # prepare an optimizable data provider
    optimizable_data_provider = RealDataOptimizablePoseProviderPose(
        dataset=dataset,
        balance=False,
    )

    #  move it to device
    optimizable_data_provider.to(device)
    optimizable_data_provider.move_images_to_device(device)
    optimizable_data_provider.viz_selection_prob(
        osp.join(log_dir, f"split_train_view_prob.png")
    )

    return optimizable_data_provider, dataset


def main_zju() -> None:
    dataset_mode = "zju"
    seq_name = "my_392"
    profile_fn = "./profiles/zju_3m.yaml"
    log_dir = f"./logs/dbg_zju/{str(uuid.uuid4())[:8]}"

    # init Gaussian Template Fitter
    solver = TGFitter(log_dir=log_dir, profile_fn=profile_fn, debug=False)

    logging.info(f"saving logs to: {log_dir}")

    # init data provider
    data_provider, _ = prepare_real_seq(seq_name=seq_name, log_dir=log_dir)

    # run optimization
    gaussian_template_model, optimized_seq = solver.run(data_provider=data_provider)
    gaussian_template_model.to(torch.device("cuda:0"))

    solver.eval_fps(gaussian_template_model, optimized_seq, rounds=10)

    viz_human_all(
        solver=solver, data_provider=optimized_seq, model=gaussian_template_model
    )

    test(
        solver,
        seq_name=seq_name,
        tto_flag=True,
        tto_step=50,
        tto_decay=10,
        dataset_mode=dataset_mode,
        pose_base_lr=3e-3,
        pose_rest_lr=3e-3,
        trans_lr=3e-3,
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main_zju()
