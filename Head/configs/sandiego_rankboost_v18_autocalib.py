# -*- coding: utf-8 -*-
"""configs/sandiego_rankboost_v18_autocalib.py

面向数据集：Sandiego

同样以 abu-airport-2 的 v18（robust_v13 + auto-pow）为模板，
并参考你提供的旧入口脚本（band=189, patch_size=11, m=30）。

Sandiego 的经验特性：
  - bands=189，谱维更长；
  - 背景材料/阴影/混合像元更复杂，容易出现“伪相似”的假警；

因此本配置的策略是：
  - 训练侧：m=30（减少 token 数，提高训练/推理速度与稳定性）
  - 推理侧：auto-pow 的背景均值目标设得更小一些（默认 0.0015），更强地压 AUC3。

运行：
  python CL_Main_LosAngeles.py --mode all  --config configs/sandiego_rankboost_v18_autocalib.py
  python CL_Main_LosAngeles.py --mode eval --config configs/sandiego_rankboost_v18_autocalib.py --test_load_weight ckpt_160_.pt

调参建议（仍然尽量只动一个参数）：
  - score_post_auto_bg_mean：建议 0.0015 起步；AUC2 掉太多就调大到 0.0018~0.0020。
"""

cfg = {
    # -------------------- I/O --------------------
    "path": "./datasets/Sandiego.mat",
    "save_dir": "./ckpt/Sandiego",
    "log_dir": "./logs/Sandiego",
    "out_dir": "./result/Sandiego",

    # -------------------- runtime --------------------
    "device": "cuda:0",
    "seed": 0,

    # -------------------- presets --------------------
    "train_preset": "robust",
    "infer_preset": "robust_v13",

    # -------------------- train --------------------
    "epoch": 160,
    "batch_size": 64,
    # m=30 token 更少，可以把 eval_batch_size 适当调大
    "eval_batch_size": 1024,

    "lr": 1e-3,
    "weight_decay": 1e-4,

    # -------------------- backbone tokenization --------------------
    # 参考旧版：m=30（band=189 时能显著减 token，训练更快更稳）
    "m": 30,
    "token_stride": None,  # 让模型内部按 m//4 自动设定 stride

    # -------------------- patch / context --------------------
    "w_size": 11,
    "patch_encode_sim": "corr",
    "patch_encode_tau": 1.0,

    # hard-negative（outer-ring）
    "hard_neg_quantile": 0.10,
    "hard_neg_region": "ring",
    "hard_neg_inner_size": 5,

    # -------------------- augment --------------------
    "use_augment": True,
    "aug_curriculum": True,
    # Sandiego 背景更复杂，可以把强增强保持满档
    "augment_view1_scale": 0.35,
    "augment_view2_scale": 1.00,

    # 模型侧谱漂移增强：略加强一点，提高 smile/漂移类鲁棒性
    "drift_p": 0.18,
    "drift_warmup": 50,

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
    # Sandiego 背景更复杂：先把背景均值锚定得更低一些，优先压 AUC3
    "score_post_auto_bg_mean": 0.00,
    "score_post_auto_pow_max": 10.0,

    # 目标占比略高于 abu-airport-2（~0.6%），排除 top 1% 更安全
    "score_post_auto_pos_ratio": 0.010,

    # 选背景子集：更严格一点（减少目标/异常混入背景校准集）
    "score_post_auto_bg_sel_q": 0.65,
    "score_post_auto_gate_low_q": 0.75,
    "score_post_auto_min_keep": 1024,

    # -------------------- sweep（可选）--------------------
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
        "score_post_sigmoid_bg_sel_q": [0.60, 0.65, 0.70],
        "score_post_sigmoid_bg_mu_q": [0.99, 0.995, 0.998],
        "score_post_sigmoid_k": [20.0, 24.0, 28.0],
        "score_post_sigmoid_min_pos_ratio": [0.005, 0.010],
        "score_post_sigmoid_rescale": [True],

        "score_post_highlift": [True],
        "score_post_highlift_thr": [0.45, 0.50, 0.55],
        "score_post_highlift_gamma": [1.5, 2.0, 2.5],
        # t0-angle extra suppression（压 AUC3 / 提 AUC5），建议 'auto'
        "score_post_t0_angle_pow": "auto",
        "score_post_t0_angle_auto_p": 0.60,
        "score_post_t0_angle_auto_thr": 0.60,
        "score_post_t0_angle_auto_top_ratio": 0.005,
    },
}
