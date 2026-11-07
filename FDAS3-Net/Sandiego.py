# Sandiego.py  (Sandiego 数据集：默认只做 SelectBest + Eval)
# =============================================================================
# 你这次报错是典型的“脚本又去训练了 + batch 太大 + 显存只有 10GB 左右”导致的 OOM：
#   - 训练阶段要保存反向传播图，Mamba/SelectiveScan 这类算子显存占用更敏感
#   - 你现在其实不需要重训，只要用已训练好的 ckpt 做 select_best + eval 即可
#
# 因此本脚本默认：
#   1) do_train=False（不重训）
#   2) eval_batch_size 独立于 batch_size（select_best/eval 用更小 batch 防止 OOM）
#   3) select_metric 默认用 auc5（更符合 Sandiego 低虚警需求）
# =============================================================================

import os

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

from typing import Dict
from CL_Train import train, select_best, eval


def main(model_config: Dict = None):
    cfg = {
        # ===== 基础配置 =====
        "epoch": 60,               # 训练时用；评估端会自动扫描 ckpt
        "band": 189,
        "batch_size": 64,          # 训练 batch（若你以后要训练，建议 32~64 起步）
        "eval_batch_size": 64,     # ★ SelectBest/Eval 推理 batch（显存紧张就 32）
        "seed": 1,
        "channel": 128,
        "lr": 1.2e-4,
        "multiplier": 1.2,
        "epision": 4,
        "grad_clip": 0.3,
        "device": "cuda:0",
        "dataset": "Sandiego",
        "path": "datasets/Sandiego.mat",
        "save_dir": "./models",
        "log_dir": "./logs",

        # ===== 评估（HTD 风格） =====
        "delta": 0.08,
        # ★ 只在 eval 内做 delta 网格选择：不重算 det，只重算 det_map，可改善低虚警区间表现
        "delta_grid": [0.05, 0.06, 0.08, 0.10, 0.12, 0.15],
        "delta_select_metric": "auc5",

        # ===== 索引顺序 =====
        "index_order": "F",
        "reshape_order": "F",

        # ===== ts 生成：Sandiego 建议集成，缓解目标谱波动 =====
        "ts_ensemble": [7, 1, 2],

# ===== 评估 profiles（不重训，专门改善 Sandiego 可视化）=====
# 关键点：
# 1) 用 PF 约束阈值生成二值 mask（最符合“目标检测”的视觉预期）
# 2) clip_pow 可视化让背景更暗、突出 top 响应
"eval_profiles": [
    {
        "name": "oracle",
        "ts_mode": "oracle",
        "postproc_mode": "clip_pow",
        "clip_p": 99.95,      # 99.9~99.99 之间调，越大越“干净”
        "pow_gamma": 8.0,     # 6~10 之间调，越大背景越暗
        "viz_pf_list": [1e-3, 1e-4],
        "save_pf_masks": True,
        "mask_min_size": 3,   # 去掉零散噪点（不影响 AUC，只影响 mask/overlay 视觉）
        "save_overlay": True,
        "overlay_gt": True,

        # ===== 额外输出：多幅“更干净”的 viridis 热力图，供你挑选 =====
        # 不影响 AUC，只改变可视化。
        # 其中 pf_mask / pf_binary 会用“背景分位数”生成 PF 约束阈值，只保留 top 响应，视觉会更像“检测结果”。
        "viridis_variants": [
            {"tag": "raw", "mode": "raw"},
            {"tag": "clip999_g6", "mode": "clip_pow", "clip_p": 99.9, "pow_gamma": 6.0},
            {"tag": "clip9995_g8", "mode": "clip_pow", "clip_p": 99.95, "pow_gamma": 8.0},

            # PF 约束 heatmap：建议优先看这几个
            {"tag": "pf1e-3_cc3", "mode": "pf_mask", "pf": 1e-3, "min_cc": 3, "rescale": True, "median": 3, "gamma": 1.0},
            {"tag": "pf5e-4_cc3", "mode": "pf_mask", "pf": 5e-4, "min_cc": 3, "rescale": True, "median": 3, "gamma": 1.0},
            {"tag": "pf1e-4_cc3", "mode": "pf_mask", "pf": 1e-4, "min_cc": 3, "rescale": True, "median": 3, "gamma": 1.0},

            # 如果你遇到“出现一大块高响应背景”的情况，可试试限制连通域最大面积（max_cc）
            {"tag": "pf1e-3_cc3_max500", "mode": "pf_mask", "pf": 1e-3, "min_cc": 3, "max_cc": 500, "rescale": True, "median": 3},

            # 二值版（最干净，适合做检测 mask 图）
            {"tag": "pf1e-3_bin", "mode": "pf_binary", "pf": 1e-3, "min_cc": 3},
        ],
    }
],


        # ===== 模型架构（不动结构） =====
        "patch_size": 7,
        "m": 14,
        "state_size": 64,
        "layer": 2,
        "use_wavelet": True,
        "wavelet_level": 2,
        "learnable_lp": True,
        "wavelet_name": "db4",
        "attn_detach": True,
        "multi_m": [4, 8, 12, 16],
        "m_gate_type": "hilo",
        "use_bg_suppress": True,

        # ===== 数据增强 / SESA（训练时用）=====
        "use_dynamic_aug": True,
        "aug_auto_calibrate": True,
        "aug_calib_strength": 0.35,
        "sesa_tau": 0.40,
        "sesa_topk": 20,
        "sesa_sigma": 1.0,
        "train_use_eval_standard": True,

        # ===== 工程扰动：谱漂移（训练时用）=====
        "use_spectral_drift": True,
        "drift_p": 0.20,
        "drift_shift_range": (-2.0, 2.0),

        # ===== 训练稳健性（仅 do_train=True 时用）=====
        "amp": False,
        "amp_dtype": "fp16",
        "use_ema": True,
        "ema_decay": 0.998,
        "balance_sampler": True,
        "pos_ratio_in_batch": 0.20,
        "log_interval": 50,
        "max_nan_steps": 10,

        # ===== BAN & 子空间约束（结构不改，参数可调）=====
        "lambda_ban": 0.30,
        "lambda_null_pos": 0.25,
        "ban_beta": 0.75,
        "ban_m_pos": 0.30,
        "ban_m_neg": 0.08,
        "train_bg_K": 8,
        "bg_proto_momentum": 0.99,

        # 困难负样本（训练时用）
        "use_hard_neg": True,
        "hard_neg_topk_start": 0.02,
        "hard_neg_topk_end": 0.10,
        "hard_neg_warmup_epochs": 35,
        "hard_neg_margin": 0.08,
        "lambda_hard_neg": 0.02,

        # ===== 可视化后处理 =====
        "postproc_mode": "clip_pow",
        "post_low_p": 0.5,
        "post_hi_p": 99.5,
        "nbs_delta": 0.03,
        "rank_gamma": 3.0,

        # auto 模式下：如果出现“稀疏图/背景抬亮”，优先回退 clip_pow（更符合检测图预期）
        "post_sparse_fallback": "clip_pow",
        "clip_p": 99.95,
        "pow_gamma": 8.0,

        # 选择最佳 ckpt 的指标：Sandiego 推荐 auc5
        "select_metric": "auc5",

        # select_best 循环里每个 ckpt 后清 cache，避免显存碎片堆积
        "empty_cache_each_ckpt": True,

        # ===== 运行控制 =====
        "do_train": False,   # ★默认不重训：只做 SelectBest + Eval
    }

    if model_config:
        cfg.update(model_config)

    # 1) （可选）训练
    if cfg.get("do_train", False):
        print("开始训练（Sandiego）...")
        train(cfg)

    # 2) SelectBest（自动扫描 ckpt_*_.pt）
    print("按 HTD‑Mamba 口径选择最佳权重 ...")
    max_score, opt_epoch = select_best(cfg)
    cfg["test_load_weight"] = f"ckpt_{opt_epoch}_.pt"
    print(f"将使用最优权重进行最终评估：{cfg['test_load_weight']}")

    # 3) Eval（输出到 result/）
    print("开始 HTD‑style 最终评估 ...")
    eval(cfg)


if __name__ == "__main__":
    print("------------------ Sandiego (SelectBest + Eval) ------------------")
    main()
