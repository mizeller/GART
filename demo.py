"""small debug script to understand the code"""

# default
import torch
import uuid
import logging

# custom
from lib_data.get_data import prepare_real_seq
from viz_utils import viz_human_all
from test_utils import test
from solver import TGFitter


def main_zju() -> None:
    dataset_mode = "zju"
    seq_name = "my_392"
    profile_fn = "./profiles/zju/zju_3m.yaml"
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
