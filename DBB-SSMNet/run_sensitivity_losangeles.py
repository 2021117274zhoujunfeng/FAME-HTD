"""Hyper-parameter sensitivity analysis on LosAngeles (abu-airport-2).

This script runs sensitivity (ablation) experiments for three key hyper-parameters:
  1) delta               : post-processing sharpness in eval() (NO retraining needed)
  2) layer               : number of Mamba blocks (requires retraining)
  3) repr_reg_weight     : representation regularizer weight (requires retraining)

Metrics are exactly the repo's AUC1~AUC5 (printed by eval()).

Usage (one command):
  python run_sensitivity_losangeles.py --device cuda:0 --epoch 200

Tips:
  - select_best() is expensive. You can speed it up with --select_stride 5 (default).
    If you want the strict original behaviour, set --select_stride 1.
  - By default, this script will NOT save detection-map images / ROC csvs for every run
    (to avoid generating tons of files). You can enable them with --save_visuals / --save_curves.
"""

import argparse
import copy
import csv
import os
from datetime import datetime

import torch

from CL_Train import train, eval, select_best


def _parse_float_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(float(x))
    return out


def _parse_int_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(int(x))
    return out


def _f2tag(v: float) -> str:
    # 0.05 -> 0p05, 1e-3 -> 0p001
    s = f"{float(v):g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _make_dataset_tag(base_dataset: str, base_model: str, layer: int, repr_w: float) -> str:
    return f"{base_dataset}_{base_model}_L{int(layer)}_R{_f2tag(repr_w)}"


def _apply_base_model_switches(cfg: dict, base_model: str):
    base_model = str(base_model).lower()
    if base_model == "dual_bi":
        cfg.update(
            {
                "use_grad_branch": True,
                "main_bidirectional": True,
                "grad_bidirectional": True,
            }
        )
    elif base_model == "dual_uni":
        cfg.update(
            {
                "use_grad_branch": True,
                "main_bidirectional": False,
                "grad_bidirectional": False,
            }
        )
    elif base_model == "main_bi":
        cfg.update(
            {
                "use_grad_branch": False,
                "main_bidirectional": True,
                "grad_bidirectional": False,
            }
        )
    elif base_model == "main_uni":
        cfg.update(
            {
                "use_grad_branch": False,
                "main_bidirectional": False,
                "grad_bidirectional": False,
            }
        )
    else:
        raise ValueError(
            "Unknown --base_model. Use one of: dual_bi, dual_uni, main_bi, main_uni"
        )
    return cfg


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
        # ---------------- Added switches (from improved version) ----------------
        "use_grad_branch": True,
        "main_bidirectional": True,
        "grad_bidirectional": True,
        "bi_fuse": "gate",
        "bi_share_params": False,
        # ---------------- extra controls ----------------
        "select_stride": 5,  # default for this script
        "result_dir": "result_sensitivity",
        "result_name": None,
        "save_visuals": False,
        "save_curves": False,
        "verbose": False,
    }


def _run_train_select_eval(cfg: dict):
    print("\n" + "=" * 90)
    print(f"[Train+Select+Eval] dataset_tag = {cfg['dataset']}")
    print(
        f"layer={cfg.get('layer')}, repr_reg_weight={cfg.get('repr_reg_weight')}, delta={cfg.get('delta')} | "
        f"use_grad_branch={cfg.get('use_grad_branch')}, main_bi={cfg.get('main_bidirectional')}, grad_bi={cfg.get('grad_bidirectional')}"
    )
    print("=" * 90)

    train(cfg)
    best_auc, best_epoch = select_best(cfg)
    cfg["test_load_weight"] = f"ckpt_{best_epoch}_.pt"

    metrics = eval(cfg)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_auc1_select"] = float(best_auc)
    return metrics


def _append_row(rows: list, sweep: str, cfg: dict, metrics: dict, extra: dict | None = None):
    row = {
        "sweep": str(sweep),
        "dataset": str(cfg.get("dataset")),
        "base_model": str(cfg.get("base_model", "")),
        "delta": float(cfg.get("delta", 0.0)),
        "layer": int(cfg.get("layer", 0)),
        "repr_reg_weight": float(cfg.get("repr_reg_weight", 0.0)),
        "best_epoch": int(metrics.get("best_epoch", -1)),
        "best_auc1_select": float(metrics.get("best_auc1_select", 0.0)),
        "AUC1": float(metrics.get("AUC1", 0.0)),
        "AUC2": float(metrics.get("AUC2", 0.0)),
        "AUC3": float(metrics.get("AUC3", 0.0)),
        "AUC4": float(metrics.get("AUC4", 0.0)),
        "AUC5": float(metrics.get("AUC5", 0.0)),
        "execution_time_sec": float(metrics.get("execution_time_sec", 0.0)),
        "memory_mb": float(metrics.get("memory_mb", 0.0)),
    }
    if extra:
        row.update(extra)
    rows.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument(
        "--base_model",
        type=str,
        default="dual_bi",
        help="dual_bi | dual_uni | main_bi | main_uni",
    )
    parser.add_argument(
        "--delta_list",
        type=str,
        default="0.003,0.006,0.008,0.010,0.012,0.015",
        help="comma-separated, e.g. 0.006,0.008,0.010",
    )
    parser.add_argument(
        "--layer_list",
        type=str,
        default="1,2,3",
        help="comma-separated ints, e.g. 1,2,3",
    )
    parser.add_argument(
        "--repr_list",
        type=str,
        default="0,0.01,0.02,0.05,0.1",
        help="comma-separated floats, e.g. 0,0.02,0.05",
    )
    parser.add_argument(
        "--select_stride",
        type=int,
        default=5,
        help="evaluate checkpoints every N epochs when selecting best (default 5)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="result_sensitivity",
        help="where to write the summary CSV (and optional visuals/curves)",
    )
    parser.add_argument("--save_visuals", action="store_true", help="save det-map images for each eval")
    parser.add_argument("--save_curves", action="store_true", help="save ROC csv files for each eval")
    parser.add_argument("--verbose_eval", action="store_true", help="print background/target statistics")
    args = parser.parse_args()

    delta_list = _parse_float_list(args.delta_list)
    layer_list = _parse_int_list(args.layer_list)
    repr_list = _parse_float_list(args.repr_list)

    base_cfg = build_base_config(args.device, args.epoch)
    base_cfg["select_stride"] = int(args.select_stride)
    base_cfg["result_dir"] = str(args.result_dir)
    base_cfg["save_visuals"] = bool(args.save_visuals)
    base_cfg["save_curves"] = bool(args.save_curves)
    base_cfg["verbose"] = bool(args.verbose_eval)

    base_cfg["base_model"] = str(args.base_model)
    _apply_base_model_switches(base_cfg, args.base_model)

    # Ensure result dir exists
    os.makedirs(base_cfg["result_dir"], exist_ok=True)

    # summary csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(base_cfg["result_dir"], f"LosAngeles_sensitivity_{timestamp}.csv")

    rows = []

    # -----------------------------------------
    # Step 0) Train baseline once (used by delta sweep and as reference)
    # -----------------------------------------
    base_layer = int(base_cfg["layer"])
    base_repr = float(base_cfg["repr_reg_weight"])
    base_dataset_name = _make_dataset_tag(base_cfg["dataset"], args.base_model, base_layer, base_repr)

    cfg_base = copy.deepcopy(base_cfg)
    cfg_base["dataset"] = base_dataset_name
    cfg_base["result_name"] = base_dataset_name  # for output prefix
    base_metrics = _run_train_select_eval(cfg_base)

    _append_row(rows, sweep="baseline", cfg=cfg_base, metrics=base_metrics)

    # -----------------------------------------
    # Step 1) delta sweep (NO retraining)
    # -----------------------------------------
    print("\n" + "#" * 90)
    print("[Sensitivity] delta sweep (no retraining)")
    print("#" * 90)

    best_epoch = int(base_metrics["best_epoch"])
    best_ckpt = f"ckpt_{best_epoch}_.pt"

    for d in delta_list:
        cfg_d = copy.deepcopy(cfg_base)
        cfg_d["delta"] = float(d)
        cfg_d["test_load_weight"] = best_ckpt
        cfg_d["result_name"] = f"{base_dataset_name}_delta{_f2tag(d)}"
        metrics_d = eval(cfg_d)
        metrics_d["best_epoch"] = best_epoch
        metrics_d["best_auc1_select"] = float(base_metrics.get("best_auc1_select", 0.0))
        _append_row(rows, sweep="delta", cfg=cfg_d, metrics=metrics_d)

    # -----------------------------------------
    # Step 2) layer sweep (retraining)
    # -----------------------------------------
    print("\n" + "#" * 90)
    print("[Sensitivity] layer sweep (retraining)")
    print("#" * 90)

    for lyr in layer_list:
        lyr = int(lyr)
        if lyr == base_layer:
            # reuse baseline
            continue

        cfg_l = copy.deepcopy(base_cfg)
        cfg_l["layer"] = lyr
        cfg_l["repr_reg_weight"] = base_repr
        cfg_l["dataset"] = _make_dataset_tag(base_cfg["dataset"], args.base_model, lyr, base_repr)
        cfg_l["result_name"] = cfg_l["dataset"]
        metrics_l = _run_train_select_eval(cfg_l)
        _append_row(rows, sweep="layer", cfg=cfg_l, metrics=metrics_l)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Add the baseline point into the layer sweep as well (so you have a complete table)
    _append_row(rows, sweep="layer", cfg=cfg_base, metrics=base_metrics, extra={"note": "reuse_baseline"})

    # -----------------------------------------
    # Step 3) repr_reg_weight sweep (retraining)
    # -----------------------------------------
    print("\n" + "#" * 90)
    print("[Sensitivity] repr_reg_weight sweep (retraining)")
    print("#" * 90)

    for rw in repr_list:
        rw = float(rw)
        if abs(rw - base_repr) < 1e-12:
            # reuse baseline
            continue

        cfg_r = copy.deepcopy(base_cfg)
        cfg_r["layer"] = base_layer
        cfg_r["repr_reg_weight"] = rw
        cfg_r["dataset"] = _make_dataset_tag(base_cfg["dataset"], args.base_model, base_layer, rw)
        cfg_r["result_name"] = cfg_r["dataset"]
        metrics_r = _run_train_select_eval(cfg_r)
        _append_row(rows, sweep="repr_reg_weight", cfg=cfg_r, metrics=metrics_r)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Add the baseline point into repr sweep
    _append_row(rows, sweep="repr_reg_weight", cfg=cfg_base, metrics=base_metrics, extra={"note": "reuse_baseline"})

    # -----------------------------------------
    # Write CSV
    # -----------------------------------------
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\n" + "=" * 90)
    print(f"Sensitivity analysis finished. Summary CSV saved to:\n  {summary_csv}")
    print("Key metrics columns: AUC1~AUC5")
    print("=" * 90)


if __name__ == "__main__":
    main()