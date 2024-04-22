"""small debug script to understand the code"""

from solver import TGFitter
import torch
from test_utils import test
from viz_utils import viz_human_all


def main_zju() -> None:
    dataset_mode = "zju"
    seq_name = "my_392"
    profile_fn = "./profiles/zju/zju_3m.yaml"
    log_dir = f"./logs/dbg_zju"

    solver = TGFitter(
        log_dir=log_dir,
        profile_fn=profile_fn,
        mode="human",
    )

    data_provider, _ = solver.prepare_real_seq(seq_name, dataset_mode, split="train")
    _, optimized_seq = solver.run(data_provider)
    solver.eval_fps(solver.load_saved_model(), optimized_seq, rounds=10)

    viz_human_all(solver, optimized_seq, training_skip=1)

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


def main_ubc() -> None:
    # CLI arguments
    device = torch.device("cuda:0")
    dataset_mode = "instant_avatar_wild"
    seq_name = "ourpose_ubc_91+bCFG1jOS"
    profile_fn = "./profiles/ubc/ubc_mlp.yaml"
    # log_dir = "./logs/ubc_mlp/seq=ourpose_ubc_91+bCFG1jOS_prof=ubc_mlp_data=instant_avatar_wild"
    log_dir = "./logs/hello_world"
    mode = "human"
    smpl_path = "./data/smpl_model/SMPL_NEUTRAL.pkl"

    # init solver
    solver = TGFitter(
        log_dir=log_dir,
        profile_fn=profile_fn,
        mode=mode,
        template_model_path=smpl_path,
        device=device,
    )

    # Optimize and eval FPS
    data_provider, trainset = solver.prepare_real_seq(
        seq_name, dataset_mode, split="train"
    )
    _, optimized_seq = solver.run(data_provider)
    solver.eval_fps(solver.load_saved_model(), optimized_seq, rounds=10)

    # Visualize
    viz_human_all(solver, optimized_seq, training_skip=1)

    # this breaks for ubc dataset
    # test(
    #     solver,
    #     seq_name=seq_name,
    #     tto_flag=True,
    #     tto_step=50,
    #     tto_decay=20,
    #     dataset_mode=dataset_mode,
    #     pose_base_lr=4e-3,
    #     pose_rest_lr=4e-3,
    #     trans_lr=4e-3,
    #     training_optimized_seq=optimized_seq,
    # )

    pass


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # main_ubc()
    main_zju()
