"""small debug script to understand the code"""

# default
import torch
import uuid
import logging

# custom
from utils.viz_utils import viz_human_all
from solver import TGFitter


from lib_data.zju_mocap import Dataset as ZJUDataset
from lib_data.data_provider import RealDataOptimizablePoseProviderPose


def main_zju() -> None:
    # init data provider
    data_provider = RealDataOptimizablePoseProviderPose(
        dataset=ZJUDataset(),
        balance=False,
    )

    # init Gaussian Template Fitter
    solver = TGFitter(log_dir=f"./logs/dbg_zju/{str(uuid.uuid4())[:8]}", debug=False)

    # run optimization
    gaussian_template_model, optimized_seq = solver.run(data_provider=data_provider)

    solver.eval_fps(gaussian_template_model, optimized_seq, rounds=10)

    viz_human_all(
        solver=solver, data_provider=optimized_seq, model=gaussian_template_model
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main_zju()
