# -*- coding: utf-8 -*-
"""configs/abu_beach4_rankboost_v18_mixbg_osp.py

面向数据集：abu-beach-4

该配置以你已在 abu-airport-2 上验证通过的 v18（infer_preset=robust_v13：robust_v12 + 自动 pow 标定）为模板，
并参考你提供的 HTD‑Mamba 旧入口脚本（band=102, patch_size=5, m=15）。

目标：
  - 重新训练以适配你的改良版模型（Bi‑Mamba + 多谱表示 + WIN/BatchHard 等）
  - 推理端尽量“免扫参”：用 auto-pow 把背景均值锚定到一个稳定水平，从而让 AUC5 更可控。

使用前请确认：
  1) 你的数据文件放在 ./datasets/abu-beach-4.mat
  2) .mat 内变量名为 data / map（你上传的数据就是这样）

运行：
  # 训练 + 自动选择最佳 ckpt + 评估
  python CL_Main_LosAngeles.py --mode all --config configs/abu_beach4_rankboost_v18_mixbg_osp.py

  # 只训练
  python CL_Main_LosAngeles.py --mode train --config configs/abu_beach4_rankboost_v18_mixbg_osp.py

  # 只评估某个 ckpt
  python CL_Main_LosAngeles.py --mode eval --config configs/abu_beach4_rankboost_v18_mixbg_osp.py --test_load_weight ckpt_160_.pt

调参建议（尽量只动一个参数）：
  - score_post_auto_bg_mean：越小背景抑制越强（AUC3 更低 -> AUC5 更高），但可能轻微牺牲 AUC2。
    推荐先用 0.0020；如果 AUC5 仍偏低，再试 0.0015。
"""

cfg = {
    # -------------------- I/O --------------------
    "path": "./datasets/abu-beach-4.mat",
    # 为了不和其它数据集混 ckpt，建议分目录
    "save_dir": "./ckpt/abu-beach-4",
    "log_dir": "./logs/abu-beach-4",
    "out_dir": "./result/abu-beach-4",

    # -------------------- runtime --------------------
    "device": "cuda:0",
    "seed": 0,

    # -------------------- presets --------------------
    "train_preset": "robust",
    "infer_preset": "robust_v13",

    # -------------------- train --------------------
    # abu-airport-2 用 160 已经够用；如果你想更充分收敛，也可改 200
    "epoch": 160,

    # 该数据 patch 小（w_size=5），显存更友好，可略增 batch（若 OOM 再降回 64）
    "batch_size": 96,
    "eval_batch_size": 1024,

    "lr": 1e-3,
    "weight_decay": 1e-4,

    # -------------------- backbone tokenization --------------------
    # 参考你旧版配置：m=15（band=102 时 15 更均衡：细粒度/速度）
    "m": 5,
    # 重要：若你在 config 中改了 m，建议把 token_stride 设为 None，
    # 让模型内部按 stride=m//4 自动计算（避免 build_cfg 里先算好的 stride 不一致）
    "token_stride": 2,

    # -------------------- patch / context --------------------
    "w_size": 9,
    "patch_encode_sim": "corr",
    "patch_encode_tau": 1.0,

    # hard-negative
    "hard_neg_quantile": 0.10,
    "hard_neg_region": "ring",
    # w=5 时 inner=3：排除中心 3x3，从外圈取 negative
    "hard_neg_inner_size": 3,

    # -------------------- augment --------------------
    "use_augment": True,
    "aug_curriculum": True,
    # beach 数据往往同质区多，增强过强易“把细谱形抹平”，这里略保守
    "augment_view1_scale": 0.30,
    "augment_view2_scale": 0.85,

    # 模型侧谱漂移增强：该数据已归一化且 bands=102，先偏保守
    "drift_p": 0.10,
    "drift_warmup": 40,

    # false-negative suppression
    "fn_topk": 10,

    # WIN + BatchHard
    "use_win": True,
    "win_thr_mode": "quantile",
    "win_thr_q": 0.10,
    "win_k": 20.0,
    "win_min_w": 1e-4,

    "w_bh": 0.06,
    "bh_margin": 0.12,
    "bh_start_epoch": 25,

    # -------------------- eval --------------------
    "eval_save_outputs": True,
    "eval_save_debug": True,
    "ts_type": 7,
        # 针对 abu-beach-4：混合像素占比高，建议开启 mixbg + bg_osp
        "mixbg_enable": True,
        "mixbg_samples": 12,
        "mixbg_alphas": [0.05, 0.10, 0.20, 0.35],
        "mixbg_angle_q": 0.60,
        "mixbg_pool_size": 4096,
        "mixbg_use_alpha_prior": True,
        "mixbg_alpha_mu": 0.18,
        "mixbg_alpha_sigma": 0.18,

        "use_bg_osp": True,
        "bg_osp_k": 6,
        "bg_osp_ridge": 1e-3,
        "bg_osp_apply": "both",  # "z_only" / "spec_only" / "both"

    # -------------------- selection --------------------
    "select_metric": "AUC5",
    "select_strategy": "auto_auc1_guard",
    "select_primary": "AUC5",
    "select_auc1_keep_ratio": 0.995,
    "select_auc1_keep_drop": 0.010,

    # ============================================================
    # ✅ 自动 pow 标定（免 sweep）
    # ============================================================
    "score_post_pow_auto": True,
    # 背景均值目标：越小抑制越强
    "score_post_auto_bg_mean": 0.0020,
    "score_post_auto_pow_max": 8.0,
    # 目标很稀疏（~0.3%），排除 top 0.5% 基本够用
    "score_post_auto_pos_ratio": 0.005,
    # 用 sim_bg/gate 选“更像背景”的子集
    "score_post_auto_bg_sel_q": 0.60,
    "score_post_auto_gate_low_q": 0.70,
    "score_post_auto_min_keep": 2048,

    # -------------------- sweep（可选，默认不需要）--------------------
    "sweep_from_select_csv": True,
    "sweep_topk_epochs": 10,
    "sweep_rank_metric": "AUC1",
    "sweep_objective": "AUC5",
    "sweep_constraints": {},

    "sweep_grid": {
        "score_post_norm": ["none", "q"],
        "score_post_norm_q": [0.995, 0.999],

        "score_post_sigmoid": [True],
        "score_post_sigmoid_mu": ["bg_q"],
        "score_post_sigmoid_bg_sel_q": [0.55, 0.60, 0.65],
        "score_post_sigmoid_bg_mu_q": [0.99, 0.995, 0.998],
        "score_post_sigmoid_k": [20.0, 24.0, 28.0],
        "score_post_sigmoid_min_pos_ratio": [0.003, 0.005, 0.010],
        "score_post_sigmoid_rescale": [True],

        # pow 由 auto 处理，网格里不扫
        "score_post_highlift": [True],
        "score_post_highlift_thr": [0.45, 0.50, 0.55],
        "score_post_highlift_gamma": [1.5, 2.0, 2.5],
    },
}
