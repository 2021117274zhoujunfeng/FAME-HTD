import time
from CL_Train import train, eval, select_best
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(model_config=None):
    # 默认配置
    modelConfig = {
        # -------- 训练与优化配置 --------
        "epoch": 200,
        "batch_size": 80,
        "lr": 1e-4,
        "multiplier": 2.0,
        "grad_clip": 1.0,
        "epision": 10,
        "seed": 1,
        # -------- 模型结构配置 --------
        "band": 189,  # Sandiego2 光谱维度 :contentReference[oaicite:8]{index=8}
        "channel": 32,  # 通道数 ↑
        "state_size": 32,  # 状态维度 ↑
        "layer": 1,
        # 原来是 patch_size = 13，这里稍微缩小上下文窗口，减弱“目标+跑道”被编码成同一正样本的程度
        "patch_size": 11,
        "m": 5,  # 维持细粒度谱分组（原本就是 5） :contentReference[oaicite:10]{index=10}
        "repr_reg_weight": 0.05,
        # -------- 检测阶段配置 --------
        # Sandiego2 背景亮点更多，为了更强的“尖锐度”再比 Sandiego 略小一档
        "delta": 0.006,
        # -------- 路径与设备配置 --------
        "device": "cuda:0",
        "training_load_weight": None,
        "save_dir": "./models/",
        "test_load_weight": "ckpt_0_.pt",
        "dataset": "Sandiego2",
        "path": "datasets/Sandiego2.mat",
    }

    if model_config is not None:
        modelConfig = model_config

    print("------------------Sandiego2-----------------------")
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
