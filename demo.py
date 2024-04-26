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


def main_zju() -> None:
    log_dir = f"./logs/dbg_zju/{str(uuid.uuid4())[:8]}"
    device = torch.device("cuda:0")

    logging.info(f"saving logs to: {log_dir}")
    logging.info("Prepare real seq: my_392")

    # init data provider
    data_provider = RealDataOptimizablePoseProviderPose(
        dataset=ZJUDataset(),
        balance=False,
    )

    #  move it to device
    data_provider.to(device)
    data_provider.move_images_to_device(device)
    data_provider.viz_selection_prob(osp.join(log_dir, f"split_train_view_prob.png"))

    # init Gaussian Template Fitter
    solver = TGFitter(log_dir=log_dir, debug=False)

    # run optimization
    gaussian_template_model, optimized_seq = solver.run(data_provider=data_provider)
    gaussian_template_model.to(device)

    solver.eval_fps(gaussian_template_model, optimized_seq, rounds=10)

    viz_human_all(
        solver=solver, data_provider=optimized_seq, model=gaussian_template_model
    )

    # test(solver, seq_name=seq_name, tto_step=50, tto_decay=10)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main_zju()
