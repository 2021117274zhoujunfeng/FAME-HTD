import time

from CL_Train import train, eval, select_best
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(model_config=None):
    modelConfig = {
        "epoch": 200,
        "band": 205,
        "batch_size": 80,
        "seed": 1,
        "channel": 32,
        "lr": 1e-4,
        "multiplier": 2.0,
        "epision": 10,
        "grad_clip": 1.0,
        "repr_reg_weight": 0.05,  # 表征分布正则权重
        "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_dir": "./models/",
        "test_load_weight": "ckpt_128_.pt",
        "patch_size": 11,
        "m": 5,
        "state_size": 32,
        "layer": 1,
        # delta 越小，检测图越“尖锐”，AUC1 不变但 AUC5 会明显增大
        "delta": 0.012,
        "dataset": "abu-airport-2",
        "path": "datasets/abu-airport-2.mat",
    }
    # if model_config is not None:
    #     modelConfig = model_config
    # Step 1: 训练模型
    print("开始训练模型...")
    train(modelConfig)

    # # Step 2: 选择最佳模型
    # print("开始选择最佳模型...")
    best_auc, best_epoch = select_best(modelConfig)
    print(f"最佳 AUC: {best_auc}, 最佳 Epoch: {best_epoch}")

    # Step 3: 加载最佳模型并进行评估
    print("开始评估最佳模型...")
    modelConfig["test_load_weight"] = f"ckpt_{best_epoch}_.pt"
    eval(modelConfig)


if __name__ == "__main__":
    print("------------------LosAngeles-----------------------")
    main()
