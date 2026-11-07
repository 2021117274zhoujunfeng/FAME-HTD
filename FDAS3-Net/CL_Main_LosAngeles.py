# CL_Main_LosAngeles.py  (train -> HTD-select_best -> HTD-eval; keep rich training logs)
import os
from typing import Dict
from CL_Train import train, select_best, eval

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(model_config: Dict = None):
    cfg = {
        # ===== 基础配置 =====
        "epoch": 100,
        "band": 205,
        "batch_size": 64,
        "seed": 1,
        "channel": 128,
        "lr": 8e-5,
        "multiplier": 1.5,
        "epision": 4,
        "grad_clip": 0.5,
        "device": "cuda:0",
        "dataset": "abu-airport-2",
        "path": "datasets/abu-airport-2.mat",
        "save_dir": "./models",
        "log_dir": "./logs",

        # ===== 评估（HTD 风格） =====
        "delta": 0.08,   # 对应 HTD 的 exp 非线性映射参数

        # ===== 索引/可视化顺序（训练数据类用） =====
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

        # ===== 数据增强（与你的安全增强链与 SESA 配置一致） =====
        "use_dynamic_aug": True,
        "aug_auto_calibrate": True,
        "aug_calib_strength": 0.65,
        "sesa_tau": 0.35,
        "sesa_topk": 25,
        "sesa_sigma": 2.0,

        # ★ 训练归一化与评估端对齐（默认 True）
        "train_use_eval_standard": True,

        # （可选）微幅谱漂移
        "use_spectral_drift": False,  # 若训练数据仪器/大气窄带漂移明显，可改 True
        "drift_p": 0.15,

        # ===== 训练稳健性 =====
        "amp": True,
        "use_ema": True,
        "ema_decay": 0.998,
        "balance_sampler": True,
        "pos_ratio_in_batch": 0.25,
        "log_interval": 50,

        # ===== BAN & 子空间约束 =====
        "lambda_ban": 0.35, "lambda_null_pos": 0.25, "ban_beta": 0.75, "ban_m_pos": 0.30, "ban_m_neg": 0.08,
        "train_bg_K": 8, "bg_proto_momentum": 0.99,

        # 困难负样本
        "use_hard_neg": True, "hard_neg_topk": 0.10, "hard_neg_margin": 0.08, "lambda_hard_neg": 0.05,

        # ===== 运行控制 =====
        "do_train": True,
    }

    if model_config:
        cfg.update(model_config)

    # 1) 训练（含详尽日志打印）
    if cfg.get("do_train", False):
        print("开始训练（BAN + 样本级门控 + EMA/AMP + 平衡采样 + 困难负样本）...")
        train(cfg)

    # 2) HTD 口径选择最佳模型
    print("按 HTD‑Mamba 口径选择最佳权重 ...")
    max_auc, opt_epoch = select_best(cfg)
    cfg["test_load_weight"] = f"ckpt_{opt_epoch}_.pt"
    print(f"将使用最优权重进行最终评估：{cfg['test_load_weight']}")

    # 3) HTD 口径最终评估（含图片与 ROC CSV 落盘到 result/）
    print("开始 HTD‑style 最终评估 ...")
    eval(cfg)

if __name__ == "__main__":
    main()
