import time
from CL_Train import train, eval, select_best
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(model_config=None):
    # 默认配置（可在调用 main(model_config=...) 时覆盖）
    modelConfig = {
        # -------- 训练与优化配置 --------
        "epoch": 200,
        "batch_size": 80,
        "lr": 1e-4,
        "multiplier": 2.0,  # warmup 最终倍数
        "grad_clip": 1.0,
        "epision": 10,
        "seed": 1,
        # -------- 模型结构配置 --------
        "band": 189,  # Sandiego 光谱维度 :contentReference[oaicite:0]{index=0}
        "channel": 32,  # 通道数 ↑，与 LosAngeles 一致 :contentReference[oaicite:1]{index=1}
        "state_size": 32,  # Mamba 状态维度 ↑
        "layer": 1,  # Mamba 层数
        "patch_size": 11,  # patch 大小保持 11（原来也是 11） :contentReference[oaicite:2]{index=2}
        "m": 5,  # 谱分组长度：从 30 改为 5，细粒度谱形建模 :contentReference[oaicite:3]{index=3}
        # 表征分布正则权重
        "repr_reg_weight": 0.05,
        # -------- 检测阶段配置 --------
        # delta 越小，检测图越“尖锐”，AUC1 基本不变，但 AUC2↓、AUC3↑
        "delta": 0.008,  # 从 0.012 略微减小 :contentReference[oaicite:4]{index=4}
        # -------- 路径与设备配置 --------
        "device": "cuda:0",
        "training_load_weight": None,
        "save_dir": "./models/",
        # 会被 select_best 自动改成 ckpt_{best_epoch}_.pt
        "test_load_weight": "ckpt_0_.pt",
        "dataset": "Sandiego",
        "path": "datasets/Sandiego.mat",
    }

    if model_config is not None:
        modelConfig = model_config

    print("------------------Sandiego-----------------------")
    print("开始训练模型...")
    train(modelConfig)

    print("开始选择最佳模型...")
    best_auc, best_epoch = select_best(modelConfig)
    print(f"最佳 AUC: {best_auc}, 最佳 Epoch: {best_epoch}")

    print("开始评估最佳模型...")
    modelConfig["test_load_weight"] = f"ckpt_{best_epoch}_.pt"
    eval(modelConfig)


if __name__ == "__main__":
    main()
