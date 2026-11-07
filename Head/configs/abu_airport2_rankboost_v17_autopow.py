# -*- coding: utf-8 -*-
"""configs/abu_airport2_rankboost_v17_autopow.py

v17（免 sweep 的 AUC5 提升方案：自动 pow 标定）

你反馈的问题：
  - 不同数据集/不同 ckpt 下，score 的分布差异很大；
  - 为了把 AUC3 压下去、把 AUC5 拉上来，需要反复扫一堆后处理参数，太麻烦。

v17 的核心思路：
  ✅ 把“压背景(AUC3)”这件事做成 *推理期自适应标定*：
     - 用 sim_bg / gate 选一批“更像背景”的像素子集；
     - 自动二分搜索 pow，使该子集的均值接近 score_post_auto_bg_mean；
     - 因为 score^pow 是严格单调映射，所以 **AUC1 不变**，
       但背景中等分数会被压向 0，从而 AUC3↓、AUC5↑。

你只需要改一个“目标背景均值”即可在不同数据集之间迁移：
  - score_post_auto_bg_mean 越小：背景抑制越强（AUC3 更低，但可能会牺牲一点 AUC2）
  - 一般建议先从 0.0020 开始；如果你追求 AUC5>200，可尝试 0.0015~0.0020

用法：
  # 训练 + 自动选 ckpt + 评估
  python CL_Main_LosAngeles.py --mode all --config configs/abu_airport2_rankboost_v17_autopow.py

  # 只评估（对已有 ckpt）
  python CL_Main_LosAngeles.py --mode eval --config configs/abu_airport2_rankboost_v17_autopow.py \
      --test_load_weight ckpt_160_.pt

说明：
  - 本配置默认使用 infer_preset=robust_v13（robust_v12 + 自动 pow）
  - 如果你想保持 robust_v11/v12 的其它设置不变，也可以仅开启：
      score_post_pow_auto=True
    并设置 score_post_auto_bg_mean / score_post_auto_pow_max 等参数。
"""

cfg = {
    # -------------------- I/O --------------------
    "path": "./datasets/abu-airport-2.mat",
    "save_dir": "./ckpt/abu-airport-2",
    "log_dir": "./logs/abu-airport-2",
    "out_dir": "./result/abu-airport-2",
    # -------------------- runtime --------------------
    "device": "cuda:0",
    "seed": 0,
    # -------------------- presets --------------------
    "train_preset": "robust",
    "infer_preset": "robust_v13",
    # -------------------- train --------------------
    "epoch": 160,
    "batch_size": 64,
    "eval_batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    # patch / context
    "w_size": 11,
    "patch_encode_sim": "corr",
    "patch_encode_tau": 1.0,
    "hard_neg_quantile": 0.10,
    # dual-window hard-negative（outer-ring）
    "hard_neg_region": "ring",
    "hard_neg_inner_size": 5,
    # augment
    "use_augment": True,
    "aug_curriculum": True,
    "augment_view1_scale": 0.35,
    "augment_view2_scale": 1.00,
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
    # 背景均值目标：越小抑制越强。想冲 AUC5>200 可尝试 0.0015~0.0020
    "score_post_auto_bg_mean": 0.0005,
    # pow 搜索上限：防止过激
    "score_post_auto_pow_max": 8.0,
    # 排除 top-K 高分像素，避免目标混入背景校准集
    "score_post_auto_pos_ratio": 0.005,
    # 用 sim_bg/gate 选背景子集的分位数
    "score_post_auto_bg_sel_q": 0.60,
    "score_post_auto_gate_low_q": 0.70,
    "score_post_auto_min_keep": 2048,
    # -------------------- sweep（可选）--------------------
    # v17 推荐先不用 sweep；如果你要 sweep，这里不要用 AUC1_min=0.99 之类的硬约束：
    # 因为单调后处理不会改变 AUC1，若 base AUC1<0.99，sweep 会被全部过滤掉导致 csv 为空。
    "sweep_from_select_csv": True,
    "sweep_topk_epochs": 10,
    "sweep_rank_metric": "AUC1",
    "sweep_objective": "AUC5",
    "sweep_constraints": {},
    # （可选）只扫少量参数，避免网格爆炸
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
        # pow 由 auto 处理，网格里就不扫了
        "score_post_highlift": [True],
        "score_post_highlift_thr": [0.45, 0.50, 0.55],
        "score_post_highlift_gamma": [1.5, 2.0, 2.5],
    },
}
