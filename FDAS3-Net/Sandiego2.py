# Sandiego2.py  (HTD‑Mamba main for Sandiego2 dataset)
import os
from typing import Dict
from CL_Train import train, select_best, eval

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(model_config: Dict = None):
    cfg = {
        # ===== 基础配置 =====
        "epoch": 100,
        "band": 189,
        "batch_size": 64,
        "seed": 1,
        "channel": 128,
        "lr": 8e-5,
        "multiplier": 1.5,
        "epision": 4,
        "grad_clip": 0.5,
        "device": "cuda:0",
        "dataset": "Sandiego2",
        "path": "datasets/Sandiego2.mat",
        "save_dir": "./models",
        "log_dir": "./logs",

        # ===== 评估（HTD 风格） =====
        "delta": 0.08,

        # ===== 索引/可视化顺序 =====
        "index_order": "F",
        "reshape_order": "F",

        # ===== 模型架构 =====
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

        # ===== 数据增强 / SESA =====
        "use_dynamic_aug": True,
        "aug_auto_calibrate": True,
        "aug_calib_strength": 0.65,
        # Sandiego2：杂波更多，邻域略收紧
        "sesa_tau": 0.46,
        "sesa_topk": 28,
        "sesa_sigma": 1.8,

        "train_use_eval_standard": True,

        # （可选）谱漂移
        "use_spectral_drift": False,
        "drift_p": 0.15,

        # ===== 训练稳健性 =====
        "amp": True,
        "use_ema": True,
        "ema_decay": 0.998,
        "balance_sampler": True,
        "pos_ratio_in_batch": 0.45,
        "log_interval": 50,

        # ===== BAN & 子空间约束 =====
        "lambda_ban": 0.35,
        "lambda_null_pos": 0.25,
        "ban_beta": 0.75,
        "ban_m_pos": 0.30,
        "ban_m_neg": 0.08,
        "train_bg_K": 4,              # 旧 cfg: num_bg_proto = 4
        "bg_proto_momentum": 0.985,

        # 困难负样本
        "use_hard_neg": True,
        "hard_neg_topk": 0.10,
        "hard_neg_margin": 0.08,
        "lambda_hard_neg": 0.05,

        "do_train": True,
    }

    if model_config:
        cfg.update(model_config)

    # 1) 训练
    if cfg.get("do_train", False):
        print("开始训练（Sandiego2，BAN + EMA/AMP + 平衡采样 + 困难负样本）...")
        train(cfg)

    # 2) HTD 口径选择最佳模型
    print("按 HTD‑Mamba 口径选择最佳权重 ...")
    max_auc, opt_epoch = select_best(cfg)
    cfg["test_load_weight"] = f"ckpt_{opt_epoch}_.pt"
    print(f"将使用最优权重进行最终评估：{cfg['test_load_weight']}")

    # 3) HTD 口径最终评估
    print("开始 HTD‑style 最终评估 ...")
    eval(cfg)


if __name__ == "__main__":
    print("------------------ Sandiego2 (HTD‑Mamba pipeline) ------------------")
    main()
