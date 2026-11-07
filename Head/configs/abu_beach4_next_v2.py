"""abu-beach-4 下一步训练配置（v2）

核心思路（针对你给的上次训练结果/可视化）：
1) beach-4 上存在“t0 角度门控/角度感知池化”反向的问题：
   真目标（桥上小目标）与 t0 的相关-余弦谱角整体偏大，而右上大块背景 confuser 反而更接近 t0，
   导致 angle_gate / angle-aware pooling 会“压真目标、抬假警”。因此这里直接关闭。
2) wavelet/deriv 的 raw 相似度在该场景也更偏向右上 confuser（你给的 debug 里 wave_sim/deriv_sim 对假警更高），
   为了先把尾部假警压住，本轮同样关闭这两路 raw-match（不影响网络内部的 wavelet/deriv 编码分支）。
3) 右上 confuser 在训练时更像“低密度/稀有背景模式”，容易不被 bg-proto 覆盖，从而在推理 bg_gate 中被当成“异常/可疑”而放行。
   这里通过：扩大 prototype bank、提高 proto_bg_quantile、降低 proto_bg_only_thr，并适度加大 w_proto，
   让背景原型更覆盖复杂背景（包含稀有背景簇），从源头提高 sim_bg 的可用性。
4) 你上次选择策略用了非常强的 AUC1 guard，导致最终落在早期 ckpt（AUC1 高但 AUC4/AUC5 更差）。
   beach-4 的曲线特性是：中后期 AUC4/AUC5 明显更优但 AUC1 会小幅回落。
   因此本轮改为直接按 AUC4 选最优 ckpt（AUC4= AUC1 + AUC2 - AUC3，本身就兼顾了排序与尾部）。

用法：
  python CL_Main_LosAngeles.py --mode all --config configs/abu_beach4_next_v2.py

注：该配置是“可直接替换”的完整 cfg dict。
"""

cfg = dict(
    # ---------------------- 基本 ----------------------
    name="abu-beach-4_next_v2",
    path="./datasets/abu-beach-4.mat",
    seed=2026,

    # 输出目录（建议单独开目录，避免覆盖历史结果）
    save_dir="./models/abu-beach-4_next_v2",
    log_dir="./logs/abu-beach-4_next_v2",
    out_dir="./result/abu-beach-4_next_v2",

    # ---------------------- 训练超参 ----------------------
    epoch=240,
    batch_size=96,
    lr=7e-4,
    weight_decay=1e-4,
    num_workers=4,

    # patch（影响 coded view / ctx / hard-neg 的构造；模型输入仍是中心像素光谱）
    w_size=15,
    patch_encode_sim="corr",
    patch_encode_tau=1.2,
    patch_encode_sigma=1.4,
    patch_encode_topk_ratio=0.20,  # 只用最相似的 20% 邻域来构造 coded view，减弱异质背景干扰

    # hard negative：ring 更适合压掉“局部相似但非同一物理像元”的干扰
    use_hard_neg=True,
    hard_neg_k=16,
    hard_neg_pool=2048,
    hard_neg_margin=0.08,
    hard_neg_start_epoch=4,
    hard_neg_quantile=0.12,
    hard_neg_region="ring",
    hard_neg_inner_size=7,

    # ---------------------- 损失权重/训练约束 ----------------------
    # 基础对比（intra / ctx）
    delta=0.07,
    delta_ctx=0.07,
    w_intra=1.0,
    w_ctx=0.30,
    ctx_thr_q=0.50,
    ctx_start_epoch=1,

    # WIN-InfoNCE：用于“背景可分簇结构”更稳定（阈值更稳）
    use_win=True,
    win_k=20,
    win_thr_mode="quantile",
    win_thr_q=0.08,
    w_win=0.22,
    win_start_epoch=3,

    # Batch-Hard：更直接地优化可排序性，防止后期 AUC1 掉得太多
    w_bh=0.10,
    bh_margin=0.14,
    bh_start_epoch=12,

    # 原型（背景聚类一致性）：更覆盖稀有背景簇以压 confuser
    w_proto=0.18,
    w_proto_ortho=0.02,
    proto_start_epoch=8,
    proto_warmup=80,
    proto_bank_size=128,
    proto_momentum=0.999,
    proto_topk=3,
    proto_temp=0.07,
    proto_density_mode="q_sigmoid",  # 用分位点做背景密度阈值，再 sigmoid 平滑
    proto_bg_quantile=0.90,
    proto_density_topk=8,
    proto_density_k=12.0,
    proto_bg_only_thr=0.35,  # 适当放宽，让稀有背景簇也能进入原型分配

    # 几何一致性：适当增强，避免表示漂到“只对某类差异敏感”
    w_geom=0.15,

    # drift 增强：保持默认强度即可（不额外加大，避免破坏 beach-4 的细粒度差异）
    drift_p=0.10,
    drift_warmup=50,

    # ---------------------- 推理/评估 ----------------------
    infer_preset="robust_v13",

    # 关键：关闭 beach-4 上会“压真目标、抬假警”的角度相关模块
    use_angle_gate=False,
    adaptive_angle_gate=False,
    angle_aware_pool=False,
    fusion_conf_use_angle=False,

    # 关键：关闭 raw-wavelet / raw-deriv 相似度证据（该场景更偏向 confuser）
    use_wavelet_match=False,
    use_deriv_match=False,
    score_w_wave=0.0,
    score_w_deriv=0.0,
    wave_sim_max_mask="none",

    # beach-4 上 angle_t0_gate 不可靠，因此所有 norm mask 都不要用 angle mask
    nbs_t_norm_mask="none",

    # 避免 quantile norm 造成大面积 clip=1 的并列最高分（top-k 会更不稳定）
    score_post_norm="max",
    score_post_norm_mask="none",

    # 背景门控适度增强（robust_v13 默认较弱，这里稍微拉回一点）
    bg_gate_thr_q=0.90,
    score_w_bg=0.55,
    bg_gate_k=28,

    # 自动化标定机制（保持开启）
    score_post_pow_auto=True,
    score_post_t0_angle_pow="auto",  # 自动触发（beach-4 这种反向 case 会自动不触发）

    # 目标 bank：保留一定的增强 + 轻量 mixbg 来覆盖混合像元，但不要过度扩大 bank
    use_target_aug=True,
    target_aug_num=8,
    target_aug_noise_std=0.004,
    target_aug_shift_range=(-2.0, 2.0),
    target_aug_scale_range=(-0.015, 0.015),
    target_aug_sine_amp_range=(0.0, 0.25),
    target_aug_sine_freq_range=(0.5, 1.5),
    target_aug_smooth_sigma_range=(0.0, 0.6),

    mixbg_enable=True,
    mixbg_samples=6,
    mixbg_alphas=[0.15, 0.35],
    mixbg_angle_q=0.55,
    mixbg_pool_size=6000,

    # ---------------------- ckpt 选择策略（关键修复） ----------------------
    # 直接按 AUC4 选最优：避免被过强 AUC1 guard 卡在早期 ckpt
    select_metric="AUC4",
    select_strategy="single",

    # 方便复现/排查：保留 best debug
    save_debug_best=True,
    eval_save_outputs=True,
    select_save_outputs=True,

    # 评估与保存频率（不必每个 epoch 都保存，2 足够；但 beach-4 曲线起伏较大，可把 eval 设为 1）
    save_interval=2,
    eval_interval=1,
)
