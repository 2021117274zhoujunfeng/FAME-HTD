# Pavia.py  (abu-beach-4)
# =============================================================================
# 你最新的结果已经说明：模型的 AUC 很好，但 detMap 可视化“很糟”主要是后处理导致的：
#   - raw det_map（参与 AUC 计算的原图）非常稀疏：背景几乎为 0，目标像元接近 1
#   - 但 rank/分位数拉伸/DSC+NBS 等“相对映射”会把占比大的弱响应背景抬亮，视觉上就会一塌糊涂
#
# 因此：这里默认只做 SelectBest + Eval（不重训），并把可视化模式改成更“绝对值域”的方式：
#   postproc_mode = 'raw'      : 直接显示 det_map（最贴近模型输出）
#   postproc_mode = 'clip_pow' : 只显示 top-分位（clip_p）并做幂次压缩（更干净、目标更突出）
#
# 你如果确认已经训练好，不要再跑训练：把 do_train 设为 False（默认已关闭）。
# =============================================================================

import os

# 这些环境变量必须在 import torch/CL_Train 之前设置才稳定生效
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 可选：减少显存碎片（尤其是你集群上显存经常被占用时）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

from typing import Dict
from CL_Train import train, select_best, eval


def main(model_config: Dict = None):
    cfg = {
        # ===== 基础配置 =====
        "epoch": 40,                 # 仅在 do_train=True 时使用；评估端会自动扫描 ckpt 文件
        "band": 102,                 # abu-beach-4: 102 bands
        "batch_size": 64,            # 训练 batch（do_train=False 时无所谓）
        "eval_batch_size": 256,      # ★ SelectBest/Eval 的推理 batch（建议单独设置，防止 OOM）
        "seed": 1,
        "channel": 128,
        "lr": 8e-5,
        "multiplier": 1.5,
        "epision": 4,
        "grad_clip": 0.5,
        "device": "cuda:0",
        "dataset": "abu-beach-4",
        "path": "datasets/abu-beach-4.mat",
        "save_dir": "./models",
        "log_dir": "./logs",

        # ===== 评估（HTD 风格）=====
        "delta": 0.08,
        # 如果你想只在评估阶段微调 delta（不重算 det，只重算 det_map），可打开：
        # "delta_grid": [0.05, 0.08, 0.10, 0.12],
        # "delta_select_metric": "auc5",

        # ===== 索引顺序 =====
        "index_order": "F",
        "reshape_order": "F",

        # ===== ts 生成 =====
        # 说明：你反馈“Pavia/abu-beach-4 指标好得夸张，不敢写”，主要原因是 oracle 会把目标谱先验用满，
        #      这本质上是一个“上限”设定。
        # 这里我们默认用更接近工程的 few-shot 设定：kshot=1（并把该像元从 eval 中剔除，避免信息泄露）。
        "ts_type": 7,
        "ts_mode": "kshot",
        "ts_k": 1,
        "ts_seed": 0,

        # ===== 评估 profiles（不重训，默认只输出 few-shot，多随机种子更可信）=====
        # 你写论文时：正文建议报告 kshot1 多 seed 的均值/方差；oracle 可作为“上限”放在附录。
        "eval_profiles": [
            {
                "name": "kshot1_s0",
                "ts_mode": "kshot",
                "ts_k": 1,
                "ts_seed": 0,
                "ts_type": 7,
                "ts_exclude_from_eval": True,
                "postproc_mode": "raw",
                "viz_pf_list": [1e-3, 1e-4],
                "save_pf_masks": True,
                "mask_min_size": 1,
                "save_overlay": True,
                "overlay_gt": True,
            },
            {
                "name": "kshot1_s1",
                "ts_mode": "kshot",
                "ts_k": 1,
                "ts_seed": 1,
                "ts_type": 7,
                "ts_exclude_from_eval": True,
                "postproc_mode": "raw",
                "viz_pf_list": [1e-3, 1e-4],
                "save_pf_masks": True,
                "mask_min_size": 1,
                "save_overlay": True,
                "overlay_gt": True,
            },
            {
                "name": "kshot1_s2",
                "ts_mode": "kshot",
                "ts_k": 1,
                "ts_seed": 2,
                "ts_type": 7,
                "ts_exclude_from_eval": True,
                "postproc_mode": "raw",
                "viz_pf_list": [1e-3, 1e-4],
                "save_pf_masks": True,
                "mask_min_size": 1,
                "save_overlay": True,
                "overlay_gt": True,
            },
            {
                "name": "kshot1_s3",
                "ts_mode": "kshot",
                "ts_k": 1,
                "ts_seed": 3,
                "ts_type": 7,
                "ts_exclude_from_eval": True,
                "postproc_mode": "raw",
                "viz_pf_list": [1e-3, 1e-4],
                "save_pf_masks": True,
                "mask_min_size": 1,
                "save_overlay": True,
                "overlay_gt": True,
            },
            {
                "name": "kshot1_s4",
                "ts_mode": "kshot",
                "ts_k": 1,
                "ts_seed": 4,
                "ts_type": 7,
                "ts_exclude_from_eval": True,
                "postproc_mode": "raw",
                "viz_pf_list": [1e-3, 1e-4],
                "save_pf_masks": True,
                "mask_min_size": 1,
                "save_overlay": True,
                "overlay_gt": True,
            },
        ],


        # ===== 模型结构超参（保持不动）=====
        "patch_size": 13,
        "m": 14,
        "state_size": 64,
        "layer": 2,
        "use_wavelet": True,
        "wavelet_level": 2,
        "learnable_lp": True,
        "wavelet_name": "db4",
        "attn_detach": True,
        "multi_m": [8, 12, 16, 20],
        "m_gate_type": "hilo",
        "use_bg_suppress": True,

        # ===== 训练稳健性（仅 do_train=True 时用）=====
        "amp": False,
        "amp_dtype": "fp16",
        "use_ema": True,
        "ema_decay": 0.998,
        "balance_sampler": True,
        "pos_ratio_in_batch": 0.25,
        "log_interval": 50,
        "max_nan_steps": 10,

        # ===== BAN & 子空间约束（参数可调，但结构不变）=====
        "lambda_ban": 0.35,
        "lambda_null_pos": 0.25,
        "ban_beta": 0.75,
        "ban_m_pos": 0.30,
        "ban_m_neg": 0.08,
        "train_bg_K": 4,
        "bg_proto_momentum": 0.985,

        # ===== 可视化后处理（★关键修复点）=====
        # 推荐 1：完全不做“相对拉伸”，直接显示 det_map
        "postproc_mode": "raw",
        "raw_gamma": 1.0,   # >1 会进一步压暗背景（可选）
        # 推荐 2：更干净的“只看 top 响应”
        # "postproc_mode": "clip_pow",
        # "clip_p": 99.9,    # 99.5~99.95 可调
        # "pow_gamma": 4.0,  # 2~6 可调

        # auto 模式下，若检测到“稀疏图/背景抬亮”，默认回退到 clip_pow
        "post_sparse_fallback": "clip_pow",

        # select_best 循环里每个 ckpt 后清一下 cache，防止显存碎片
        "empty_cache_each_ckpt": True,

        # ===== 运行控制 =====
        "do_train": False,  # ★默认关闭训练：只做 SelectBest + Eval
    }

    if model_config:
        cfg.update(model_config)

    # 1) （可选）训练
    if cfg.get("do_train", False):
        print("开始训练（abu-beach-4）...")
        train(cfg)

    # 2) 选择最优权重
    print("按 HTD‑Mamba 口径选择最佳权重 ...")
    max_score, opt_epoch = select_best(cfg)
    cfg["test_load_weight"] = f"ckpt_{opt_epoch}_.pt"
    print(f"将使用最优权重进行最终评估：{cfg['test_load_weight']}")

    # 3) 最终评估（含 detMap 图片与 ROC CSV 输出到 result/）
    print("开始 HTD‑style 最终评估 ...")
    eval(cfg)


if __name__ == "__main__":
    print("------------------ abu-beach-4 (SelectBest + Eval) ------------------")
    main()
