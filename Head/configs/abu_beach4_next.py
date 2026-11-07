# -*- coding: utf-8 -*-
"""
abu-beach-4 下一步推荐配置（仅改数据/训练参数，不改模型结构）
====================================================

核心思路（对应你当前可视化/npz 里暴露的问题）：
1) 结构性假警主要来自 sim_bg 极低的“背景 outlier”（原型没覆盖 -> bg_gate 认为是非背景）
2) beach-4 上 raw 谱角门控对真目标不友好（angle_gate_t0 真目标整体偏低、反而假警偏高）
3) wavelet 频域证据在 beach-4 上更像“假警放大器”（假警 wave_sim 明显高于真目标）

因此：
- 训练侧：扩大/放宽背景原型覆盖 + 更强排序约束（batch-hard）+ 更保守的 ctx 约束
- 推理侧：关闭 angle gate/angle-aware pooling + 关闭 wavelet match
          + 用更稳健的 target 谱选取(ts_type=0) + 采用 robust_v8(主攻结构性假警/AUC3↓)

用法：
python CL_Main_LosAngeles.py --mode all --config next_configs/abu_beach4_next.py
"""

cfg = {
    # ---------------- 必填：数据路径 ----------------
    "path": "datasets/abu-beach-4.mat",

    # checkpoint 选择策略：建议优先保证排序(AUC1)与综合(AUC4)稳定，再看 AUC5
    "select_metric": "AUC4",
    "select_strategy": "auto_auc1_guard",

    # ---------------- 训练/推理预设 ----------------
    # train_preset=robust：不改结构，只用下方 overrides 调参
    "train_preset": "robust",
    # robust_v8：MT-CAMF + 更保守 bank pooling + ASC(单调标定) + 空间抑制
    "infer_preset": "robust_v8",

    # ---------------- 训练 schedule ----------------
    "seed": 1,
    "epoch": 240,
    # beach-4 更需要 batch 内多样负样本（建议显存允许就加大）
    "batch_size": 96,
    "eval_batch_size": 4096,

    # ---------------- Patch / hard-negative（数据采样侧） ----------------
    # 更大的局部上下文，有助于学习“结构性背景”的稳定表征
    "w_size": 15,
    # dual-window outer-ring：负样本从外环取，降低同材质假负样本概率
    "hard_neg_region": "ring",
    "hard_neg_inner_size": 7,

    # patch-coded view：只用 top-k 相似像素做编码，避免边界混合导致的表示塌缩
    "patch_encode_topk_ratio": 0.25,

    # ---------------- 对比学习 false-negative 控制 ----------------
    # 过大的 topk 会把“可分离但相近”的背景材料全部屏蔽掉，beach-4 更容易掉点
    "fn_topk": 4,

    # ---------------- WIN-InfoNCE（可选：谨慎开启） ----------------
    # beach-4 上 raw 谱角并不能可靠区分真目标 vs 假警，所以 WIN 只做“极小角度”的保护
    "use_win": True,
    "win_thr_mode": "quantile",
    "win_thr_q": 0.02,   # 只保护最相似的 2% negatives
    "win_k": 16.0,

    # ---------------- Batch-hard ranking（强烈建议开启） ----------------
    # 直接提升“排序能力”（AUC1），对结构性 outlier 假警更敏感
    "w_bh": 0.12,
    "bh_margin": 0.20,
    "bh_start_epoch": 10,

    # ---------------- Context 约束（更保守） ----------------
    # beach-4 的 patch 更异质，ctx 约束过强会把不同材料拉近
    "w_ctx": 0.25,
    "ctx_thr_mode": "quantile",
    "ctx_thr_q": 0.45,

    # ---------------- 背景原型（解决 sim_bg 覆盖不足） ----------------
    "proto_bank_size": 128,
    # 放宽“可用于更新原型”的背景范围，让更多结构性背景被吸进原型
    "proto_bg_quantile": 0.85,
    # 让原型更“愿意”覆盖多样背景
    "w_proto": 0.20,
    "w_proto_ortho": 0.02,

    # ---------------- 数据增强强度（更温和，保留细粒度谱形） ----------------
    "use_augment": True,
    "aug_curriculum": True,
    "augment_view1_scale": 0.25,
    "augment_view2_scale": 0.70,
    # 适度降低会“破坏窄谱保真”的增强概率
    "augment_ctrl": {
        "drop_p": 0.10,
        "smooth_p": 0.10,
        "jitter_p": 0.10,
        "mix_p": 0.05,
        "stripe_p": 0.10,
        "stripe_amp": 0.04,
        "smile_p": 0.10,
        "keystone_p": 0.08,
        "smile_amp": 0.08,
        "keystone_amp": 0.08,
        "drift_p": 0.10,
        "drift_curriculum": True,
        "drift_warmup": 50,
        "drift_max_shift": 2,
    },

    # ---------------- 目标谱选取（推理期） ----------------
    # beach-4 上用“最接近均值的单像素”(type=7) 容易被极少数异常 target 像素带偏
    # 改成 type=0(直接均值谱) 更稳健；如果你后续想更尖锐可尝试 type=4(谱形中位数)
    "ts_type": 0,

    # ---------------- 推理侧：关闭会误导 beach-4 的项 ----------------
    "use_angle_gate": False,
    "angle_aware_pool": False,
    # wavelet 在 beach-4 上对假警更友好，直接关掉
    "use_wavelet_match": False,

    # NBS 标定时不要用 angle mask（否则高分 outlier 会污染 t_max）
    "nbs_t_norm_mask": "none",
    "wave_sim_max_mask": "none",

    # 额外：保存最优 ckpt 的 debug，便于你复盘
    "save_debug_best": True,
}
