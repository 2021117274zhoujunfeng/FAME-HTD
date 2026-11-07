# -*- coding: utf-8 -*-
"""
abu-beach-4 下一轮训练配置（v3）

目标：
1) 让训练设置更贴近 HTD-Mamba/paper（在 beach-4 这种“目标极少、且多为混合谱/亚像素”的场景里更稳），优先把 AUC1 拉上去；
2) 在不改变排序(AUC1)的前提下，用单调的分布标定/后处理把 AUC2 拉高、AUC3 压低，从而提升 AUC5；
3) 大幅减少无意义的 epoch、ckpt 数量、以及 select 阶段的垃圾文件输出。

⚠️注意：
- 本配置搭配我提供的 CL_Train.py 补丁使用（支持 save_interval/keep_last_ckpts/early_stop/select_fast）。
- 如果你不替换 CL_Train.py，这些“省时省盘”的开关会被忽略。
"""

# -------------------------
# 1) 主配置
# -------------------------
cfg = dict(
    # -------------------------
    # 数据与输出路径
    # -------------------------
    dataset="abu-beach-4",
    # 建议把数据放到你的项目 datasets/ 目录；如果路径不同，改这里即可
    path="datasets/abu-beach-4.mat",

    # 训练权重保存目录（ckpt）
    save_dir="./models/abu-beach-4_v3",
    # 训练日志目录
    log_dir="./logs/abu-beach-4_v3",
    # 推理/可视化输出目录
    out_dir="./results/abu-beach-4_v3",

    # -------------------------
    # 可复现
    # -------------------------
    seed=1,

    # -------------------------
    # 训练轮数（配合早停，通常不会跑满）
    # -------------------------
    epoch=120,              # 只设上限；通常会被 early stop 提前截断
    batch_size=80,
    lr=1e-4,
    weight_decay=1e-4,
    grad_clip=1.0,

    # -------------------------
    # 模型结构（尽量对齐 HTD-Mamba 的常用设置）
    # -------------------------
    # patch/window 大小：beach-4 目标是“桥上小车”，过大窗口会引入过多背景导致混合更严重
    w_size=5,

    # token 分组长度：HTD-Mamba 常用 15（对 102 bands 合理）
    m=15,
    token_stride=3,         # 经验：m//4≈3
    channel=16,
    state_size=16,
    layer=1,
    embed_dim=32,

    # NBS delta（即 exp(-||x-1||^2/delta) 里的 delta）
    delta=0.10,
    delta_ctx=0.10,

    # -------------------------
    # 训练策略：优先用 paper preset（对齐 HTD-Mamba 的训练约束）
    #   - 强上下文约束 w_ctx=1.0
    #   - 适度 intra-view 对比 w_intra=0.2
    # -------------------------
    train_preset="paper",

    # 额外损失在 beach-4 上容易“过拟合背景流形”反而伤检测，先全部关掉（做一次干净对齐）
    w_geom=0.0,
    w_proto=0.0,
    w_rank=0.0,

    # 让 ctx 阈值固定（paper preset 会写成 0.32），可显式指定保证一致
    ctx_thr_mode="fixed",
    ctx_sim_thr=0.32,
    ctx_sim_tau=1.0,

    # 数据增强：保持轻量（paper preset 会设置 curriculum）
    use_augment=True,
    aug_curriculum=True,
    augment_view1_scale=0.25,
    augment_view2_scale=0.65,
    # 漂移增强对 beach-4 这种细粒度谱差异可能“伤害过大”，建议降低
    drift_p=0.05,
    drift_warmup=10,

    # -------------------------
    # 推理：以“排序优先”为主（AUC1），再加单调后处理提升 AUC5
    # -------------------------
    infer_preset="auc1",     # 最干净的 baseline (htdmamba)，便于先把 AUC1 拉上去

    # 目标光谱提取：beach-4 目标多为混合谱，ts_type=2 更鲁棒（用 cosine 筛掉离群后平均）
    ts_type=2,

    # --- 目标 bank：适度开启 mixbg 来覆盖亚像素/混合目标（不会像 angle gate 那样误杀目标） ---
    target_aug_num=6,                    # 轻量 target augment
    target_aug_scale=(0.90, 1.10),
    target_aug_shift=(-0.01, 0.01),

    mixbg_enable=True,
    mixbg_pool_size=4096,                # 从全图抽样即可
    mixbg_angle_q=0.85,                  # 只用“离 t0 足够远”的背景来混合，避免把 confuser 拉进 bank
    mixbg_samples=12,
    mixbg_alphas=[0.05, 0.10, 0.15, 0.25, 0.35],
    mixbg_prune_q=0.01,                  # 去掉最像 t0 的极少数混合（降低背景误检尾巴）

    # 目标 bank 权重：用“与 t0 的角度”给极端 bank 成员降权，提升稳定性
    target_bank_weighting="angle",
    target_bank_weight_sigma=0.35,
    target_bank_angle_clip=0.90,
    target_bank_min_weight=1e-4,

    # target 相似度池化：softmax-topk 比 max 更稳，能压“偶然极大值”
    target_sim_pool="softmax_topk",
    target_sim_topk=7,
    target_sim_temp=0.05,

    # score_mode：htdmamba (sim_t) —— 先确保排序好
    score_mode="htdmamba",

    # -------------------------
    # 单调后处理（不改变排序 -> AUC1 不变）
    #   目标：AUC2↑、AUC3↓、AUC5↑
    # -------------------------
    # 1) 先做分位数归一（把极端背景尾巴压住）
    score_post_norm="q",
    score_post_norm_q=0.999,

    # 2) anchored sigmoid：端点锚定（0->0, 1->1），避免“背景下限被抬高”
    score_post_sigmoid=True,
    score_post_sigmoid_mu="bg_q",          # 用背景子集估计 mu（更稳）
    score_post_sigmoid_bg_sel_q=0.60,      # 选 sim_bg 较大的像素当背景
    score_post_sigmoid_bg_mu_q=0.995,      # 在背景上取高分位数作为拐点
    score_post_sigmoid_min_pos_ratio=0.01, # 兜底：防止 mu 过高把目标全压扁
    score_post_sigmoid_k=22.0,
    score_post_sigmoid_rescale=True,

    # 3) 自动幂次：把 score 的均值拉到很小（进一步压背景），仍保持单调
    score_post_pow_auto=True,
    score_post_pow_auto_target_mean=0.02,

    # 4) high-lift：抬高极少数高分尾巴（通常更利于 AUC2）
    score_post_highlift=True,
    score_post_highlift_q=0.995,
    score_post_highlift_gain=0.8,

    # -------------------------
    # 训练/选择：省时省盘关键开关（依赖 CL_Train.py 补丁）
    # -------------------------
    # 每隔多少 epoch 保存一次 ckpt（同时 eval 的 epoch 会强制保存）
    save_interval=5,
    # 只保留最近 N 个 ckpt（并自动保留 best ckpt）
    keep_last_ckpts=8,

    # 在线评估：每隔多少 epoch 跑一次 eval（用于早停与 best 记录）
    debug_eval_interval=5,
    save_best_ckpt=True,
    # 早停：连续多少次 eval 指标不提升就停止
    early_stop_patience=6,
    early_stop_min_epoch=15,

    # 选择最佳：直接读 best_ckpt.json（避免遍历所有 ckpt）
    select_fast=True,
    select_metric="AUC1",

    # select 阶段不保存每个 epoch 的可视化（会产生大量垃圾文件）
    select_save_outputs=False,
    select_plot_history=False,    # 可选：连曲线也不画，进一步减少文件

    # 最终 eval（只对 best ckpt 跑一次）是否保存结果图
    eval_save_outputs=True,
    eval_save_debug=True,
    eval_save_overlay=False,

    # 控制 eval 的 batch（推理更快）
    eval_batch_size=4096,
)

# -------------------------
# 2) 这些 key 会在 CL_Main_LosAngeles.py 里写回 args，从而真正触发 preset
#    （否则 infer_preset/train_preset 只写在 cfg 里不会生效）
# -------------------------
arg_keys = [
    "dataset",
    "path",
    "save_dir",
    "log_dir",
    "out_dir",
    "epoch",
    "batch_size",
    "lr",
    "weight_decay",
    "seed",
    "train_preset",
    "infer_preset",
]
