# -*- coding: utf-8 -*-
"""configs/sandiego2_rankboost_v17_autopow.py

面向数据集：Sandiego2

同样以 abu-airport-2 的 v17（robust_v13 + auto-pow）为模板，
并参考你提供的旧入口脚本（band=189, patch_size=13, m=5）。

Sandiego2 的经验特性：
  - bands=189，但旧版用 m=5：更细粒度谱形，通常 AUC1 更有优势；
  - patch=13：局部上下文更大，异质背景混入的风险更高；
  - 目标占比相对更高（你上传数据中约 1.34%），因此 auto-pow 的背景校准更容易被目标污染。

因此本配置的策略是：
  - 训练侧：保留 m=5（细粒度谱形），但适当降低 batch / eval_batch_size 防 OOM；
  - hard-negative ring：inner=7（w=13 时排除更大同质邻域，negative 更“真负”）；
  - auto-pow：更严格地排除 top 高分像素（pos_ratio=0.03），并提高 bg_sel_q/gate_low_q。

运行：
  python CL_Main_LosAngeles.py --mode all  --config configs/sandiego2_rankboost_v17_autopow.py
  python CL_Main_LosAngeles.py --mode eval --config configs/sandiego2_rankboost_v17_autopow.py --test_load_weight ckpt_160_.pt

调参建议（仍然尽量只动一个参数）：
  - score_post_auto_bg_mean：默认 0.0018；
    若 AUC5 偏低 -> 试 0.0015；若 AUC2 掉太多 -> 试 0.0020。
"""

cfg = {
    # -------------------- I/O --------------------
    "path": "./datasets/Sandiego2.mat",
    "save_dir": "./ckpt/Sandiego2",
    "log_dir": "./logs/Sandiego2",
    "out_dir": "./result/Sandiego2",

    # -------------------- runtime --------------------
    "device": "cuda:0",
    "seed": 0,

    # -------------------- presets --------------------
    "train_preset": "robust",
    "infer_preset": "robust_v13",

    # -------------------- train --------------------
    "epoch": 160,
    # m=5 tokens 很多 + w_size=13 patch 大：更容易 OOM，所以 batch 先保守
    "batch_size": 48,
    "eval_batch_size": 256,

    "lr": 1e-3,
    "weight_decay": 1e-4,

    # -------------------- backbone tokenization --------------------
    # 参考旧版：m=5（更细粒度谱形）
    "m": 5,
    "token_stride": None,

    # -------------------- patch / context --------------------
    "w_size": 13,
    "patch_encode_sim": "corr",
    "patch_encode_tau": 1.0,

    # hard-negative（outer-ring）
    "hard_neg_quantile": 0.10,
    "hard_neg_region": "ring",
    # w=13 时 inner=7：排除中心 7x7，同质邻域更大，negative 更“难但更真”
    "hard_neg_inner_size": 7,

    # -------------------- augment --------------------
    "use_augment": True,
    "aug_curriculum": True,
    "augment_view1_scale": 0.35,
    "augment_view2_scale": 1.00,

    # 模型侧谱漂移增强：中等偏强（该数据目标更“密”，过强可能伤 AUC2，先取折中）
    "drift_p": 0.20,
    "drift_warmup": 60,

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
    # Sandiego2：目标更多，先用略温和的 0.0018，避免把 AUC2 压塌；想冲 AUC5 再往 0.0015 调。
    "score_post_auto_bg_mean": 0.0005,
    "score_post_auto_pow_max": 12.0,

    # 目标占比更高（~1.34%）：排除更高比例的 top-score，降低目标污染背景校准集的概率
    "score_post_auto_pos_ratio": 0.030,

    # 背景子集选择更严格：让校准更“纯背景”
    "score_post_auto_bg_sel_q": 0.70,
    "score_post_auto_gate_low_q": 0.80,
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
        "score_post_sigmoid_bg_sel_q": [0.65, 0.70, 0.75],
        "score_post_sigmoid_bg_mu_q": [0.99, 0.995, 0.998],
        "score_post_sigmoid_k": [20.0, 24.0, 28.0],
        "score_post_sigmoid_min_pos_ratio": [0.010, 0.020],
        "score_post_sigmoid_rescale": [True],

        "score_post_highlift": [True],
        "score_post_highlift_thr": [0.45, 0.50, 0.55],
        "score_post_highlift_gamma": [1.5, 2.0, 2.5],
    },
}
