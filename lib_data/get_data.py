import sys, os.path as osp, logging, torch

sys.path.append(osp.dirname(osp.abspath(__file__)))
from zju_mocap import Dataset as ZJUDataset
from data_provider import RealDataOptimizablePoseProviderPose


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
