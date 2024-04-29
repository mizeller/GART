"""small debug script to understand the code"""

import torch
import uuid
from utils.viz_utils import viz_human_all
from solver import TGFitter


def main_zju() -> None:
    trainer = TGFitter(log_dir=f"./logs/{str(uuid.uuid4())[:8]}")
    gt_model = trainer.run()  # optimization loop
    trainer.eval_fps(gt_model, trainer.data_provider, rounds=10)
    viz_human_all(
        solver=trainer,
        data_provider=trainer.data_provider,
        model=gt_model,
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main_zju()
