"""Run LosAngeles ablation experiments.

This script will run 3 ablation settings on the LosAngeles (abu-airport-2) dataset:
  1) Main branch only + Bidirectional SSM  (Original branch, Bi)
  2) Main branch only + Unidirectional SSM (Original branch, Uni)
  3) Dual branches (Main+Grad) + Unidirectional SSM

Metrics printed by `eval()` are AUC1~AUC5 (already in the repo).

Usage:
  python run_ablation_losangeles.py --device cuda:0 --epoch 200

Notes:
  - Each experiment uses an unique `dataset` name so that checkpoints/results
    won't overwrite each other.
  - Training is time-consuming; this script is intended to be launched on a GPU.
"""

import argparse
import copy
import torch

from CL_Train import train, eval, select_best


def build_base_config(device: str, epoch: int):
    # Same defaults as CL_Main_LosAngeles.py
    return {
        "epoch": int(epoch),
        "band": 205,
        "batch_size": 80,
        "seed": 1,
        "channel": 32,
        "lr": 1e-4,
        "multiplier": 2.0,
        "epision": 10,
        "grad_clip": 1.0,
        "repr_reg_weight": 0.05,
        "device": device,
        "training_load_weight": None,
        "save_dir": "./models/",
        "test_load_weight": "ckpt_128_.pt",
        "patch_size": 11,
        "m": 5,
        "state_size": 32,
        "layer": 1,
        "delta": 0.012,
        "dataset": "abu-airport-2",
        "path": "datasets/abu-airport-2.mat",
        # ---------------- Added switches ----------------
        # default behaviour of your current improved version:
        "use_grad_branch": True,
        "main_bidirectional": False,
        "grad_bidirectional": False,
        "bi_fuse": "gate",
        "bi_share_params": False,
    }


def run_one_exp(cfg: dict):
    print("\n" + "=" * 80)
    print(f"[Running] {cfg['dataset']}")
    print(
        f"use_grad_branch={cfg.get('use_grad_branch')}, "
        f"main_bidirectional={cfg.get('main_bidirectional')}, "
        f"grad_bidirectional={cfg.get('grad_bidirectional')}, "
        f"bi_fuse={cfg.get('bi_fuse')}"
    )
    print("=" * 80)

    train(cfg)

    best_auc, best_epoch = select_best(cfg)
    print(f"[Best] AUC1={best_auc} @ epoch={best_epoch}")

    cfg["test_load_weight"] = f"ckpt_{best_epoch}_.pt"
    eval(cfg)

    # avoid GPU memory fragmentation between experiments
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epoch", type=int, default=200)
    args = parser.parse_args()

    base = build_base_config(args.device, args.epoch)

    experiments = [
        # 1) Only main branch (bidirectional)
        (
            "Abl1_MainOnly_Bi",
            {
                "use_grad_branch": False,
                "main_bidirectional": True,
                "grad_bidirectional": False,
            },
        ),
        # 2) Only main branch (unidirectional)
        (
            "Abl2_MainOnly_Uni",
            {
                "use_grad_branch": False,
                "main_bidirectional": False,
                "grad_bidirectional": False,
            },
        ),
        # 3) Dual branches (unidirectional)
        (
            "Abl3_Dual_Uni",
            {
                "use_grad_branch": True,
                "main_bidirectional": False,
                "grad_bidirectional": False,
            },
        ),
    ]

    for suffix, override in experiments:
        cfg = copy.deepcopy(base)
        cfg.update(override)
        cfg["dataset"] = f"{base['dataset']}_{suffix}"
        run_one_exp(cfg)


if __name__ == "__main__":
    main()
