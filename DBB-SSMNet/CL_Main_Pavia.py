# CL_Main_Pavia_v7.py —— abu-beach-4 专用调参版（v7）
#
# 目标：在不改模型结构的前提下，解决 abu-beach-4 的“海岸线/混合像元主导”问题：
#   - 进一步把 AUC1 拉向 0.99+
#   - 同时让 AUC5 (AUC3/AUC2) 尽量接近 Sandiego/LosAngeles 的水平
#
# 你这次最新结果（abu-beach-4roc_3d.csv 对应）：
#   AUC1=0.9789, AUC2≈0.0077, AUC3≈0.3516, AUC5≈45.6
# 现象是：背景压下去了（AUC2 变小），但有一部分目标像元得分非常低（导致 AUC1 上不去、AUC3 偏低）。
#
# v7 思路（仅调“超参配置”，不改 CL_Model / CL_Train / Data_Patch 结构）：
#   1) 适度“软化”对比学习（temperature ↑）：
#      v6 温度偏硬会把同一类目标的多模态谱形拉得太开 → 单原型检测时出现一批目标得分接近 0。
#   2) 降低 repr_reg_weight（↓）：
#      你的 VICReg 风格正则在极端前景稀少的场景里，容易被背景主导，导致目标簇被压扁/分裂。
#   3) 缩小 patch_size（11 -> 7）：
#      减少 coded_vector 被海岸线邻域“背景主成分”拖走的概率，提升正样本视角纯度。
#   4) 仍保持细谱 token + 足够容量（m=5, channel/state_size=32, layer=2）。
#
# 使用：
#   python CL_Main_Pavia_v7.py
#
# 训练完成后，会自动 select_best（按 AUC1）并做 delta sweep（只影响 AUC2/3/5，不影响 AUC1）。

import os
from CL_Train import train, eval, select_best

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(model_config=None):
    modelConfig = {
        # -------- Training --------
        "epoch": 200,
        "batch_size": 80,
        "seed": 1,
        "lr": 1e-4,
        "multiplier": 2.0,
        "grad_clip": 1.0,
        "epision": 10,

        # 对比学习：比 v6 更“软”一点，避免把目标内部多模态过度拉散
        "temperature": 0.14,
        # 表征分布正则：降低一些，避免背景主导下的过强“白化”
        "repr_reg_weight": 0.02,

        # -------- Model --------
        "band": 102,
        "channel": 32,
        "state_size": 32,
        # 保留 2 层（如果你显存紧张，可改回 1）
        "layer": 2,
        # 细粒度谱 token
        "m": 5,

        # -------- Data (positive view) --------
        "patch_size": 7,

        # -------- Eval --------
        # ts_type 会影响 AUC1（原型谱），建议优先从 7 开始（更“代表性”）
        "ts_type": 7,
        # delta 只影响映射“尖锐度”（AUC2/3/5），AUC1 基本不变；这里会 sweep
        "delta": 0.010,

        # -------- I/O --------
        "device": "cuda:0",
        "training_load_weight": None,
        "save_dir": "./models/",
        "test_load_weight": "ckpt_0_.pt",
        "dataset": "abu-beach-4",
        "path": "datasets/abu-beach-4.mat",
    }

    if model_config is not None:
        modelConfig.update(model_config)

    print("------------------Pavia / abu-beach-4 (tuned v7)-----------------------")

    # Step 1: train
    print("开始训练模型...")
    train(modelConfig)

    # Step 2: select best epoch by AUC1
    print("开始选择最佳模型（按 AUC1）...")
    best_auc, best_epoch = select_best(modelConfig)
    print(f"最佳 AUC: {best_auc}, 最佳 Epoch: {best_epoch}")

    # Step 3: eval + delta sweep (only affects AUC2/3/5)
    modelConfig["test_load_weight"] = f"ckpt_{best_epoch}_.pt"
    print("开始评估最佳模型（delta sweep）...")

    # 先覆盖常见范围；如果你发现 AUC2 仍偏大，可以把更小的 delta（0.003~0.006）也加进去
    delta_list = [0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.030]
    for d in delta_list:
        modelConfig["delta"] = float(d)
        print(f"\n--- Eval with delta = {d} ---")
        eval(modelConfig)

    print("\n[Done] v7 training + eval sweep finished.")


if __name__ == "__main__":
    main()
