#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_ablation.py

FDAS3‑Net / HTD‑Mamba 改良版 —— 消融实验一键脚本。

脚本逻辑（每个消融配置都会做一次完整流程）：
  1) （可选）训练 train(cfg)
  2) 按 HTD‑Mamba 口径 select_best(cfg)
  3) 从 result/<exp>_select_auc.csv 里读取 best epoch 对应的 AUC1~AUC5
  4) （可选）eval(cfg) 生成 detMap/ROC/可视化
  5) 汇总输出：result/ablation_summary_<dataset>.csv 与 .md

说明
----
- 不修改你现有的 train/select_best/eval 实现，属于“新增脚本即用”。
- 默认 suite=paper：覆盖你论文贡献点（小波解耦、样本自适应多尺度、数据侧稳健增强、训练评估对齐）。

"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# 你的工程函数
from CL_Train import train, select_best, eval as htd_eval


# ------------------------- helpers -------------------------

def slugify(s: str) -> str:
    """Make a filesystem-friendly name."""
    s = str(s)
    s = s.strip().replace(" ", "-")
    s = re.sub(r"[^0-9a-zA-Z_\-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_best_auc_from_select_csv(csv_path: str, opt_epoch: int) -> Dict[str, float]:
    """Parse result/<exp>_select_auc.csv written by select_best()."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"select CSV not found: {csv_path}")

    arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if arr.size == 0:
        raise RuntimeError(f"select CSV is empty: {csv_path}")
    if arr.ndim == 1:
        arr = arr[None, :]

    # header: epoch,auc1,auc2,auc3,auc4,auc5,score,eval_bs
    epochs = arr[:, 0].astype(int)
    idxs = np.where(epochs == int(opt_epoch))[0]
    if idxs.size == 0:
        # fallback: take max score row
        idx = int(np.argmax(arr[:, 6]))
    else:
        idx = int(idxs[0])

    return {
        "epoch": float(arr[idx, 0]),
        "auc1": float(arr[idx, 1]),
        "auc2": float(arr[idx, 2]),
        "auc3": float(arr[idx, 3]),
        "auc4": float(arr[idx, 4]),
        "auc5": float(arr[idx, 5]),
        "score": float(arr[idx, 6]),
        "eval_bs": float(arr[idx, 7]) if arr.shape[1] > 7 else float("nan"),
    }


def format_md_table(rows: List[Dict[str, Any]], cols: List[Tuple[str, str]]) -> str:
    """rows: list of dict; cols: [(key, header), ...]"""
    headers = [h for _, h in cols]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        vals = []
        for k, _ in cols:
            v = r.get(k, "")
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("NaN")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


@dataclass
class AblationVariant:
    name: str
    desc: str
    overrides: Dict[str, Any]


def build_base_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    """Default base config mirrors CL_Main_LosAngeles.py."""
    cfg = {
        # ===== Base =====
        "epoch": int(args.epochs),
        "band": int(args.band),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "channel": int(args.channel),
        "lr": float(args.lr),
        "multiplier": float(args.multiplier),
        "epision": int(args.epision),
        "grad_clip": float(args.grad_clip),
        "device": str(args.device),
        "dataset": str(args.dataset),
        "path": str(args.path),
        "save_dir": str(args.save_dir),
        "log_dir": str(args.log_dir),

        # ===== HTD eval mapping =====
        "delta": float(args.delta),

        # ===== index/reshape =====
        "index_order": str(args.index_order),
        "reshape_order": str(args.reshape_order),

        # ===== model =====
        "patch_size": int(args.patch_size),
        "m": int(args.m),
        "state_size": int(args.state_size),
        "layer": int(args.layer),
        "use_wavelet": True,
        "wavelet_level": int(args.wavelet_level),
        "learnable_lp": bool(args.learnable_lp),
        "wavelet_name": str(args.wavelet_name),
        "attn_detach": bool(args.attn_detach),
        "multi_m": [int(x) for x in args.multi_m],
        "m_gate_type": str(args.m_gate_type),
        "use_bg_suppress": bool(args.use_bg_suppress),

        # ===== data aug =====
        "use_dynamic_aug": bool(args.use_dynamic_aug),
        "aug_auto_calibrate": bool(args.aug_auto_calibrate),
        "aug_calib_strength": float(args.aug_calib_strength),
        "sesa_tau": float(args.sesa_tau),
        "sesa_topk": int(args.sesa_topk),
        "sesa_sigma": float(args.sesa_sigma),
        "train_use_eval_standard": bool(args.train_use_eval_standard),

        "use_spectral_drift": bool(args.use_spectral_drift),
        "drift_p": float(args.drift_p),

        # ===== training stability =====
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "use_ema": bool(args.use_ema),
        "ema_decay": float(args.ema_decay),
        "balance_sampler": bool(args.balance_sampler),
        "pos_ratio_in_batch": float(args.pos_ratio_in_batch),
        "log_interval": int(args.log_interval),

        # ===== BAN =====
        "lambda_ban": float(args.lambda_ban),
        "lambda_null_pos": float(args.lambda_null_pos),
        "ban_beta": float(args.ban_beta),
        "ban_m_pos": float(args.ban_m_pos),
        "ban_m_neg": float(args.ban_m_neg),
        "train_bg_K": int(args.train_bg_K),
        "bg_proto_momentum": float(args.bg_proto_momentum),

        # ===== hard negatives =====
        "use_hard_neg": bool(args.use_hard_neg),
        "hard_neg_topk_start": float(args.hard_neg_topk_start),
        "hard_neg_topk_end": float(args.hard_neg_topk_end),
        "hard_neg_warmup_epochs": int(args.hard_neg_warmup_epochs),
        "hard_neg_margin": float(args.hard_neg_margin),
        "lambda_hard_neg": float(args.lambda_hard_neg),

        # ===== run control =====
        "do_train": True,

        # select_best logging
        "select_print_each": bool(args.select_print_each),
        "select_save_csv": True,
        "select_out_dir": str(args.select_out_dir),
        "select_metric": str(args.select_metric),
        "eval_batch_size": int(args.eval_batch_size) if args.eval_batch_size is not None else int(args.batch_size),
    }
    return cfg


def get_suite_variants(base_cfg: Dict[str, Any], suite: str) -> List[AblationVariant]:
    """Return ablation variants for a given suite."""

    suite = str(suite).lower().strip()

    # ---- Core suite for paper contributions ----
    # 设计思想：
    #   - 频域：Wavelet 解耦 / Multi-scale / Sample-adaptive gate 分开验证
    #   - 数据侧：Robust augmentation / Train-Eval 对齐 分开验证
    if suite in ("paper", "core", "default"):
        mm = list(base_cfg.get("multi_m", [base_cfg.get("m", 14)]))
        m0 = int(base_cfg.get("m", 14))

        return [
            AblationVariant(
                name="FULL",
                desc="完整方法：Wavelet + Multi-scale + Adaptive gate + RobustAug + AlignNorm",
                overrides=dict(
                    use_wavelet=True,
                    multi_m=mm,
                    m_gate_type="hilo",
                    use_dynamic_aug=True,
                    train_use_eval_standard=True,
                ),
            ),
            AblationVariant(
                name="NO_ADAPT_GATE",
                desc="去掉样本自适应门控（保持 Multi-scale，但 gate 变成常量融合）",
                overrides=dict(
                    use_wavelet=True,
                    multi_m=mm,
                    m_gate_type="const",
                ),
            ),
            AblationVariant(
                name="NO_WAVELET",
                desc="去掉小波解耦（为避免耦合，gate 同时固定为 const；保留 Multi-scale）",
                overrides=dict(
                    use_wavelet=False,
                    multi_m=mm,
                    m_gate_type="const",
                ),
            ),
            AblationVariant(
                name="SINGLE_SCALE",
                desc="去掉 Multi-scale（只保留单尺度 m；gate 固定 const）",
                overrides=dict(
                    use_wavelet=True,
                    multi_m=[m0],
                    m_gate_type="const",
                ),
            ),
            AblationVariant(
                name="NO_ROBUST_AUG",
                desc="去掉数据侧稳健增强（use_dynamic_aug=False）",
                overrides=dict(
                    use_dynamic_aug=False,
                    aug_auto_calibrate=False,
                    use_spectral_drift=False,
                ),
            ),
            AblationVariant(
                name="NO_ALIGN_NORM",
                desc="取消训练/评估归一化口径对齐（train_use_eval_standard=False）",
                overrides=dict(
                    train_use_eval_standard=False,
                ),
            ),
        ]

    # ---- Extended suite (optional extra knobs) ----
    if suite in ("extended", "ext"):
        core = get_suite_variants(base_cfg, "paper")
        mm = list(base_cfg.get("multi_m", [base_cfg.get("m", 14)]))
        return core + [
            AblationVariant(
                name="FIXED_WAVELET",
                desc="Wavelet 滤波器固定（learnable_lp=False）",
                overrides=dict(
                    use_wavelet=True,
                    learnable_lp=False,
                    multi_m=mm,
                    m_gate_type="hilo",
                ),
            ),
            AblationVariant(
                name="WAVELET_L1",
                desc="Wavelet 分解层数=1（wavelet_level=1）",
                overrides=dict(
                    use_wavelet=True,
                    wavelet_level=1,
                    multi_m=mm,
                    m_gate_type="hilo",
                ),
            ),
            AblationVariant(
                name="NO_BG_SUPPRESS",
                desc="去掉 BackgroundSuppressor（use_bg_suppress=False）",
                overrides=dict(
                    use_bg_suppress=False,
                ),
            ),
        ]

    raise ValueError(f"Unknown suite: {suite}. Use --suite paper|extended")


def maybe_train(cfg: Dict[str, Any], *, force_train: bool = False) -> bool:
    """Train if needed. Return True if training executed."""
    ckpt_dir = os.path.join(cfg["save_dir"], cfg["dataset"])
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*_.pt")))
    if (not force_train) and (len(ckpts) > 0):
        print(f"[Ablation] Skip training (found {len(ckpts)} ckpts) -> {ckpt_dir}")
        return False

    print(f"[Ablation] Train -> exp={cfg['dataset']}  (save_dir={ckpt_dir})")
    train(cfg)
    return True


def run_one_variant(
    base_cfg: Dict[str, Any],
    variant: AblationVariant,
    *,
    seed: int,
    force_train: bool,
    do_eval: bool,
) -> Dict[str, Any]:
    """Run train->select_best->(optional eval) for one variant & one seed."""

    cfg = copy.deepcopy(base_cfg)
    cfg.update(variant.overrides)

    # seed isolation
    cfg["seed"] = int(seed)

    # experiment tag
    base_dataset = str(base_cfg["dataset"])
    exp_name = slugify(f"{base_dataset}__{variant.name}__s{seed}")
    cfg["dataset"] = exp_name

    ensure_dir(cfg.get("select_out_dir", "result"))

    t0 = time.time()
    maybe_train(cfg, force_train=force_train)

    # select best
    print(f"[Ablation] select_best -> {cfg['dataset']}")
    best_score, opt_epoch = select_best(cfg)

    # load best metrics from select CSV
    csv_path = os.path.join(cfg.get("select_out_dir", "result"), f"{cfg['dataset']}_select_auc.csv")
    best_auc = read_best_auc_from_select_csv(csv_path, opt_epoch)

    # optional eval (for figures/ROC)
    if do_eval:
        cfg["test_load_weight"] = f"ckpt_{opt_epoch}_.pt"
        print(f"[Ablation] eval -> {cfg['dataset']}  weight={cfg['test_load_weight']}")
        htd_eval(cfg)

    dt = time.time() - t0

    row = {
        "variant": variant.name,
        "seed": int(seed),
        "exp": cfg["dataset"],
        "opt_epoch": int(opt_epoch),
        "select_score": float(best_score),
        "auc1": float(best_auc["auc1"]),
        "auc2": float(best_auc["auc2"]),
        "auc3": float(best_auc["auc3"]),
        "auc4": float(best_auc["auc4"]),
        "auc5": float(best_auc["auc5"]),
        "time_sec": float(dt),
        "desc": variant.desc,
    }
    return row


def aggregate_by_variant(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate rows across multiple seeds: mean±std."""
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)

    out = []
    for v, rs in by.items():
        def _stat(key: str) -> Tuple[float, float]:
            xs = np.array([float(x[key]) for x in rs], dtype=np.float64)
            return float(xs.mean()), float(xs.std(ddof=0))

        m1, s1 = _stat("auc1")
        m2, s2 = _stat("auc2")
        m3, s3 = _stat("auc3")
        m4, s4 = _stat("auc4")
        m5, s5 = _stat("auc5")

        out.append({
            "variant": v,
            "nseed": len(rs),
            "auc1_mean": m1, "auc1_std": s1,
            "auc2_mean": m2, "auc2_std": s2,
            "auc3_mean": m3, "auc3_std": s3,
            "auc4_mean": m4, "auc4_std": s4,
            "auc5_mean": m5, "auc5_std": s5,
        })

    # keep a stable order (FULL first if exists)
    def _rank(name: str) -> int:
        order = ["FULL", "NO_ADAPT_GATE", "NO_WAVELET", "SINGLE_SCALE", "NO_ROBUST_AUG", "NO_ALIGN_NORM"]
        if name in order:
            return order.index(name)
        return 999

    out.sort(key=lambda x: (_rank(x["variant"]), x["variant"]))
    return out


def save_outputs(
    *,
    raw_rows: List[Dict[str, Any]],
    agg_rows: List[Dict[str, Any]],
    out_dir: str,
    tag: str,
):
    ensure_dir(out_dir)

    # raw CSV
    raw_csv = os.path.join(out_dir, f"ablation_raw_{tag}.csv")
    with open(raw_csv, "w", encoding="utf-8") as f:
        keys = list(raw_rows[0].keys()) if raw_rows else []
        f.write(",".join(keys) + "\n")
        for r in raw_rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    # aggregate CSV
    agg_csv = os.path.join(out_dir, f"ablation_summary_{tag}.csv")
    with open(agg_csv, "w", encoding="utf-8") as f:
        keys = list(agg_rows[0].keys()) if agg_rows else []
        f.write(",".join(keys) + "\n")
        for r in agg_rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    # markdown table
    md_path = os.path.join(out_dir, f"ablation_summary_{tag}.md")
    md_rows = []
    for r in agg_rows:
        md_rows.append({
            "Variant": r["variant"],
            "Seeds": r["nseed"],
            "AUC1": f"{r['auc1_mean']:.4f}±{r['auc1_std']:.4f}",
            "AUC2": f"{r['auc2_mean']:.4f}±{r['auc2_std']:.4f}",
            "AUC3": f"{r['auc3_mean']:.4f}±{r['auc3_std']:.4f}",
            "AUC4": f"{r['auc4_mean']:.4f}±{r['auc4_std']:.4f}",
            "AUC5": f"{r['auc5_mean']:.4f}±{r['auc5_std']:.4f}",
        })

    # manual markdown (string fields)
    cols = [("Variant", "Variant"), ("Seeds", "Seeds"), ("AUC1", "AUC1"), ("AUC2", "AUC2"), ("AUC3", "AUC3"), ("AUC4", "AUC4"), ("AUC5", "AUC5")]
    md = format_md_table(md_rows, cols)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n[Ablation] Saved:")
    print(f"  raw   -> {raw_csv}")
    print(f"  sum   -> {agg_csv}")
    print(f"  table -> {md_path}")
    print("\n--- Markdown Table ---")
    print(md)


# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FDAS3‑Net ablation runner (AUC1~AUC5)")

    # dataset
    p.add_argument("--dataset", type=str, default="abu-airport-2", help="base dataset name (used as tag)")
    p.add_argument("--path", type=str, default="datasets/abu-airport-2.mat", help="path to .mat (contains data/map)")
    p.add_argument("--band", type=int, default=205)

    # runtime
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save_dir", type=str, default="./models")
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--select_out_dir", type=str, default="result")

    # suite & control
    p.add_argument("--suite", type=str, default="paper", choices=["paper", "extended"], help="which ablation suite to run")
    p.add_argument("--seeds", type=int, nargs="*", default=[1], help="seeds to run (e.g., --seeds 1 2 3)")
    p.add_argument("--force_train", action="store_true", help="force training even if ckpts exist")
    p.add_argument("--no_eval", action="store_true", help="skip eval() (only train + select_best + table)")

    # train hyperparams
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--channel", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--multiplier", type=float, default=1.5)
    p.add_argument("--epision", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=0.5)

    # HTD mapping param
    p.add_argument("--delta", type=float, default=0.08)

    # index order
    p.add_argument("--index_order", type=str, default="F")
    p.add_argument("--reshape_order", type=str, default="F")

    # model arch
    p.add_argument("--patch_size", type=int, default=13)
    p.add_argument("--m", type=int, default=14)
    p.add_argument("--state_size", type=int, default=64)
    p.add_argument("--layer", type=int, default=2)

    p.add_argument("--wavelet_level", type=int, default=2)
    p.add_argument("--learnable_lp", action="store_true", default=True)
    p.add_argument("--wavelet_name", type=str, default="db4")
    p.add_argument("--attn_detach", action="store_true", default=True)

    p.add_argument("--multi_m", type=int, nargs="+", default=[8, 12, 16, 20])
    p.add_argument("--m_gate_type", type=str, default="hilo")
    p.add_argument("--use_bg_suppress", action="store_true", default=True)

    # data aug & alignment
    p.add_argument("--use_dynamic_aug", action="store_true", default=True)
    p.add_argument("--aug_auto_calibrate", action="store_true", default=True)
    p.add_argument("--aug_calib_strength", type=float, default=0.65)
    p.add_argument("--sesa_tau", type=float, default=0.35)
    p.add_argument("--sesa_topk", type=int, default=25)
    p.add_argument("--sesa_sigma", type=float, default=2.0)
    p.add_argument("--train_use_eval_standard", action="store_true", default=True)

    p.add_argument("--use_spectral_drift", action="store_true", default=False)
    p.add_argument("--drift_p", type=float, default=0.15)

    # training stability
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16", "bfloat16"])
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.998)
    p.add_argument("--balance_sampler", action="store_true", default=True)
    p.add_argument("--pos_ratio_in_batch", type=float, default=0.25)
    p.add_argument("--log_interval", type=int, default=50)

    # BAN
    p.add_argument("--lambda_ban", type=float, default=0.35)
    p.add_argument("--lambda_null_pos", type=float, default=0.25)
    p.add_argument("--ban_beta", type=float, default=0.75)
    p.add_argument("--ban_m_pos", type=float, default=0.30)
    p.add_argument("--ban_m_neg", type=float, default=0.08)
    p.add_argument("--train_bg_K", type=int, default=8)
    p.add_argument("--bg_proto_momentum", type=float, default=0.99)

    # hard neg
    p.add_argument("--use_hard_neg", action="store_true", default=True)
    p.add_argument("--hard_neg_topk_start", type=float, default=0.05)
    p.add_argument("--hard_neg_topk_end", type=float, default=0.15)
    p.add_argument("--hard_neg_warmup_epochs", type=int, default=20)
    p.add_argument("--hard_neg_margin", type=float, default=0.08)
    p.add_argument("--lambda_hard_neg", type=float, default=0.05)

    # select/eval
    p.add_argument("--select_print_each", action="store_true", default=True)
    p.add_argument("--select_metric", type=str, default="auc1", choices=["auc1", "auc3", "auc4", "auc5"])
    p.add_argument("--eval_batch_size", type=int, default=None, help="eval batch size for select_best/eval; default=batch_size")

    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = build_base_cfg(args)
    variants = get_suite_variants(base_cfg, args.suite)

    raw_rows: List[Dict[str, Any]] = []

    print("\n================ Ablation Plan ================")
    for v in variants:
        print(f"- {v.name}: {v.desc}")
    print("================================================\n")

    for v in variants:
        for sd in args.seeds:
            print("\n" + "=" * 80)
            print(f"[Run] variant={v.name} | seed={sd}")
            print("=" * 80)
            row = run_one_variant(
                base_cfg,
                v,
                seed=int(sd),
                force_train=bool(args.force_train),
                do_eval=(not args.no_eval),
            )
            raw_rows.append(row)

    agg_rows = aggregate_by_variant(raw_rows)

    tag = slugify(str(args.dataset))
    save_outputs(raw_rows=raw_rows, agg_rows=agg_rows, out_dir=args.select_out_dir, tag=tag)


if __name__ == "__main__":
    main()
