# -*- coding: utf-8 -*-
"""
Sandiego 下一步推荐配置（仅改数据/训练参数，不改模型结构）
====================================================

你当前 Sandiego 的主要问题（从 debug_ckpt_56_.npz/可视化看）：
- 真目标 sim_t 高、angle_raw_t0 小（物理一致性强）
- 少量假警的 sim_bg 也很低（原型没覆盖 -> bg_gate 认为是“非背景”），导致 τ-Pf 尾部抬起(AUC3↑)

因此策略：
- 训练侧：适度扩大原型覆盖（让这些 outlier 背景被建模）
- 推理侧：保留 angle gate，并略增强其权重；采用 robust_v8（主攻结构性 outlier 假警）

用法：
python CL_Main_LosAngeles.py --mode all --config next_configs/sandiego_next.py
"""
print("Sandiego")
cfg = {
    "path": "datasets/Sandiego.mat",

    "select_metric": "AUC4",
    "select_strategy": "auto_auc1_guard",

    "train_preset": "robust",
    "infer_preset": "robust_v8",

    "seed": 1,
    "epoch": 220,
    "batch_size": 80,
    "eval_batch_size": 4096,

    # ---------------- 原型：让“低 sim_bg 的背景 outlier”更容易被覆盖 ----------------
    "proto_bank_size": 96,
    "proto_bg_quantile": 0.80,
    "w_proto": 0.18,
    "w_proto_ortho": 0.02,

    # ---------------- 排序能力（轻量开启） ----------------
    "w_bh": 0.08,
    "bh_margin": 0.15,
    "bh_start_epoch": 15,

    # ---------------- 角度门控：Sandiego 上是有效证据，略增强 ----------------
    "use_angle_gate": True,
    "angle_gate_thr": 0.30,   # rad
    "angle_gate_k": 14.0,
    "angle_gate_w": 1.30,
    "angle_aware_pool": True,

    # 保持 deriv 证据（有助于抑制频域噪声）
    "use_deriv_match": True,
    "use_wavelet_match": True,

    # 目标谱选取：Sandiego 上 type=7 通常没问题
    "ts_type": 7,

    "save_debug_best": True,
}
