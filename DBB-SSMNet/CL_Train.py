import os
import time
from typing import Dict
import torch
import numpy as np
from ts_generation import ts_generation, checkFile, standard
from Data_Patch import Data
import matplotlib.pyplot as plt
from CL_Model import SpectralGroupAttention
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from Scheduler import GradualWarmupScheduler
import torch.nn.functional as F
import scipy.io as sio
from sklearn import metrics
import random
from datetime import datetime


def seed_torch(seed=1):
    """
    Keep the seed fixed thus the results can keep stable
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def paintTrend(losslist, epochs=100, stride=10):
    result_dir = "result"  # 定义结果保存目录
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)  # 创建目录如果不存在
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(result_dir, f"{timestamp}_Loss.png")
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.title("loss-trend")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(np.arange(0, epochs, stride))
    plt.xlim(0, epochs)
    plt.plot(losslist, color="r")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()


def train(modelConfig: Dict):
    seed_torch(modelConfig["seed"])
    device = torch.device(modelConfig["device"])
    dataset = Data(modelConfig["path"], w_size=modelConfig["patch_size"])
    dataloader = DataLoader(
        dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    # model setup
    net_model = SpectralGroupAttention(
        band=modelConfig["band"],
        group_length=modelConfig["m"],
        channel_dim=modelConfig["channel"],
        state_size=modelConfig["state_size"],
        device=device,
        layer=modelConfig["layer"],
        # --- ablation switches (default keeps the original behaviour) ---
        use_grad_branch=bool(modelConfig.get("use_grad_branch", True)),
        main_bidirectional=bool(modelConfig.get("main_bidirectional", False)),
        grad_bidirectional=bool(modelConfig.get("grad_bidirectional", False)),
        bi_fuse=modelConfig.get("bi_fuse", "gate"),
        bi_share_params=bool(modelConfig.get("bi_share_params", False)),
    ).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(
            torch.load(
                os.path.join(
                    modelConfig["save_dir"], modelConfig["training_load_weight"]
                ),
                map_location=device,
            ),
            strict=False,
        )
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4
    )
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler,
    )
    path = modelConfig["save_dir"] + "/" + modelConfig["dataset"] + "/"
    checkFile(path)
    # start training
    net_model.train()
    loss_list = []
    for e in range(modelConfig["epoch"]):
        epoch_loss = 0.0
        num_iter = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for center_pixels, coded_vectors in tqdmDataLoader:
                # train
                combined_vectors = torch.cat([center_pixels, coded_vectors], dim=0)
                optimizer.zero_grad()
                x_0 = combined_vectors.to(device)
                features = net_model(x_0)

                # 对比学习损失
                loss_contrast = info_nce_loss(
                    features,
                    modelConfig["batch_size"],
                    temperature=float(modelConfig.get("temperature", 0.1)),
                )

                # 表征分布正则（VICReg 风格），仅依赖无标签特征
                reg_weight = float(modelConfig.get("repr_reg_weight", 0.05))
                if reg_weight > 0.0:
                    loss_reg = representation_regularizer(
                        features, modelConfig["batch_size"]
                    )
                    loss = loss_contrast + reg_weight * loss_reg
                else:
                    loss = loss_contrast

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"]
                )
                optimizer.step()
                epoch_loss += loss.item()
                num_iter += 1
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": e,
                        "loss: ": loss.item(),
                        "LR": optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )
        warmUpScheduler.step()
        avg_epoch_loss = epoch_loss / max(1, num_iter)
        loss_list.append(avg_epoch_loss)
        if e % 1 == 0:
            torch.save(
                net_model.state_dict(), os.path.join(path, "ckpt_" + str(e) + "_.pt")
            )
    paintTrend(loss_list, epochs=modelConfig["epoch"], stride=20)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def cosin_similarity(x, y):
    assert x.shape[1] == y.shape[1]
    x_norm = np.sqrt(np.sum(x**2, axis=1))
    y_norm = np.sqrt(np.sum(y**2, axis=1))
    x_y = np.sum(np.multiply(x, y), axis=1)
    return x_y / (x_norm * y_norm + 1e-8)


def transpose(x):
    return x.transpose(-2, -1)


def info_nce_loss(x, size, temperature=0.1, reduction="mean", symmetric=True):
    """
    对称 InfoNCE：
    - 正样本：center_i <-> coded_i
    - 负样本：同 batch 内其它样本
    同时计算 x0->x1 和 x1->x0 两个方向的 CE loss 取平均，
    可以让两个 view 都学到更稳定的特征。
    """
    batch_x0 = x[:size]  # center
    batch_x1 = x[size:]  # coded

    batch_x0, batch_x1 = normalize(batch_x0, batch_x1)

    # x0 -> x1
    logits01 = batch_x0 @ transpose(batch_x1)
    labels = torch.arange(len(batch_x0), device=batch_x0.device)
    loss01 = F.cross_entropy(logits01 / temperature, labels, reduction=reduction)

    if not symmetric:
        return loss01

    # x1 -> x0
    logits10 = batch_x1 @ transpose(batch_x0)
    loss10 = F.cross_entropy(logits10 / temperature, labels, reduction=reduction)

    return 0.5 * (loss01 + loss10)


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return the off-diagonal elements of a square matrix as a flat vector."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    避免特征维度坍缩，鼓励每一维 std >= 1。
    只依赖 batch 内的无标签特征。
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """
    惩罚不同特征维度之间的线性相关性，
    使得各维度更独立，有利于阈值的稳定。
    """
    n, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    return _off_diagonal(cov).pow(2).sum() / d


def representation_regularizer(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    VICReg 风格的表征正则：variance + covariance。
    - 不使用目标标签或先验，只约束整体特征分布。
    - 在背景居多的场景下，有利于让背景特征集中、
      同时保持各维度有足够的方差。
    """
    z0 = x[:size]
    z1 = x[size:]

    v_loss = variance_loss(z0) + variance_loss(z1)
    c_loss = covariance_loss(z0) + covariance_loss(z1)

    # 协方差项稍微减弱，避免梯度过大
    return v_loss + 0.01 * c_loss


def select_best(modelConfig: Dict):
    """Select the best checkpoint by AUC1 (ROC AUC).

    Notes
    -----
    The original implementation evaluated every epoch checkpoint, which is very
    expensive for sensitivity/ablation sweeps. This version keeps the same
    metric (AUC1) but supports a configurable stride:

        modelConfig['select_stride'] = 1  # default, evaluate every epoch
        modelConfig['select_stride'] = 5  # evaluate every 5 epochs

    It also caches dataset preprocessing (data_matrix/labels/target prior) to
    avoid redundant work inside the epoch loop.
    """

    seed_torch(modelConfig["seed"])
    device = torch.device(modelConfig["device"])
    opt_epoch = 0
    max_auc = 0.0
    path = modelConfig["save_dir"] + "/" + modelConfig["dataset"] + "/"

    stride = int(modelConfig.get("select_stride", 1))
    if stride < 1:
        stride = 1

    # --------- cache dataset & target prior once ---------
    mat = sio.loadmat(modelConfig["path"])
    data = mat["data"]
    map = mat["map"]
    data = standard(data)
    ts_type = int(modelConfig.get("ts_type", 7))
    target_spectrum = ts_generation(data, map, ts_type)

    h, w, c = data.shape
    numpixel = h * w
    data_matrix = np.reshape(data, [-1, c], order="F")
    data_tensor = torch.from_numpy(data_matrix)  # keep on CPU

    y_l = np.reshape(map, [-1, 1], order="F")

    target_prior = torch.from_numpy(target_spectrum.T).to(device)
    target_prior = torch.unsqueeze(target_prior, dim=1)  # [1,1,Band] (or [B,1,Band])

    batch_size = int(modelConfig["batch_size"])

    # evaluate epochs with stride, always include the last epoch
    epochs = list(range(0, int(modelConfig["epoch"]), stride))
    last_e = int(modelConfig["epoch"]) - 1
    if last_e not in epochs:
        epochs.append(last_e)
    epochs = sorted(set([e for e in epochs if e >= 0]))

    for e in epochs:
        ckpt_path = os.path.join(path, f"ckpt_{e}_.pt")
        if not os.path.exists(ckpt_path):
            continue

        with torch.no_grad():
            model = SpectralGroupAttention(
                band=modelConfig["band"],
                group_length=modelConfig["m"],
                channel_dim=modelConfig["channel"],
                state_size=modelConfig["state_size"],
                device=device,
                layer=modelConfig["layer"],
                # --- ablation switches ---
                use_grad_branch=bool(modelConfig.get("use_grad_branch", True)),
                main_bidirectional=bool(modelConfig.get("main_bidirectional", False)),
                grad_bidirectional=bool(modelConfig.get("grad_bidirectional", False)),
                bi_fuse=modelConfig.get("bi_fuse", "gate"),
                bi_share_params=bool(modelConfig.get("bi_share_params", False)),
            ).to(device)

            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt)
            model.eval()

            detection_map = np.zeros([numpixel], dtype=np.float64)

            target_features = model(target_prior).cpu().detach().numpy()

            # batch inference
            for i in range(0, numpixel - batch_size, batch_size):
                pixels = data_tensor[i : i + batch_size].to(device)
                pixels = torch.unsqueeze(pixels, dim=1)
                features = model(pixels).cpu().detach().numpy()
                detection_map[i : i + batch_size] = cosin_similarity(
                    features, target_features
                )

            left_num = numpixel % batch_size
            if left_num != 0:
                pixels = data_tensor[-left_num:].to(device)
                pixels = torch.unsqueeze(pixels, dim=1)
                features = model(pixels).cpu().detach().numpy()
                detection_map[-left_num:] = cosin_similarity(features, target_features)

            detection_map = np.reshape(detection_map, [h, w], order="F")
            detection_map = standard(detection_map)
            detection_map = np.clip(detection_map, 0, 1)

            y_p = np.reshape(detection_map, [-1, 1], order="F")

            # calculate AUC1
            try:
                fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
                fpr = fpr[1:]
                tpr = tpr[1:]
                auc = round(metrics.auc(fpr, tpr), modelConfig["epision"])
            except Exception:
                auc = 0.5

            if auc > max_auc:
                max_auc = float(auc)
                opt_epoch = int(e)

    print(f"最佳 Epoch: {opt_epoch}")
    return max_auc, opt_epoch

def eval(modelConfig: Dict):
    """Evaluate a checkpoint and report AUC1~AUC5.

    This function keeps the original printing/saving behaviour, but adds:
      - Optional output controls:
            result_dir   : output folder (default 'result')
            result_name  : output file prefix (default modelConfig['dataset'])
            save_visuals : whether to save detection map images (default True)
            save_curves  : whether to save ROC csv files (default True)
            verbose      : whether to print background/target stats (default True)
      - Returns a dict with the computed metrics.
    """
    start_time = time.perf_counter()

    seed_torch(modelConfig["seed"])
    device = torch.device(modelConfig["device"])
    path = modelConfig["save_dir"] + "/" + modelConfig["dataset"] + "/"

    # -------- output controls --------
    result_dir = modelConfig.get("result_dir", "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    save_visuals = bool(modelConfig.get("save_visuals", True))
    save_curves = bool(modelConfig.get("save_curves", True))
    verbose = bool(modelConfig.get("verbose", True))

    # file prefix
    Name = modelConfig.get("result_name", modelConfig["dataset"])

    with torch.no_grad():
        mat = sio.loadmat(modelConfig["path"])
        data = mat["data"]
        map = mat["map"]
        data = standard(data)
        ts_type = int(modelConfig.get("ts_type", 7))
        target_spectrum = ts_generation(data, map, ts_type)
        h, w, c = data.shape
        numpixel = h * w
        data_matrix = np.reshape(data, [-1, c], order="F")

        model = SpectralGroupAttention(
            band=modelConfig["band"],
            group_length=modelConfig["m"],
            channel_dim=modelConfig["channel"],
            state_size=modelConfig["state_size"],
            device=device,
            layer=modelConfig["layer"],
            # --- ablation switches ---
            use_grad_branch=bool(modelConfig.get("use_grad_branch", True)),
            main_bidirectional=bool(modelConfig.get("main_bidirectional", False)),
            grad_bidirectional=bool(modelConfig.get("grad_bidirectional", False)),
            bi_fuse=modelConfig.get("bi_fuse", "gate"),
            bi_share_params=bool(modelConfig.get("bi_share_params", False)),
        ).to(device)

        ckpt = torch.load(
            os.path.join(path, modelConfig["test_load_weight"]), map_location=device
        )
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        batch_size = int(modelConfig["batch_size"])
        detection_map = np.zeros([numpixel], dtype=np.float64)

        target_prior = torch.from_numpy(target_spectrum.T).to(device)
        target_prior = torch.unsqueeze(target_prior, dim=1)
        target_features = model(target_prior).cpu().detach().numpy()

        # batch inference
        for i in range(0, numpixel - batch_size, batch_size):
            pixels = data_matrix[i : i + batch_size]
            pixels = torch.from_numpy(pixels).to(device)
            pixels = torch.unsqueeze(pixels, dim=1)
            features = model(pixels).cpu().detach().numpy()
            detection_map[i : i + batch_size] = cosin_similarity(
                features, target_features
            )

        left_num = numpixel % batch_size
        if left_num != 0:
            pixels = data_matrix[-left_num:]
            pixels = torch.from_numpy(pixels).to(device)
            pixels = torch.unsqueeze(pixels, dim=1)
            features = model(pixels).cpu().detach().numpy()
            detection_map[-left_num:] = cosin_similarity(features, target_features)

        detection_map = detection_map.astype(np.float64)
        detection_map = np.clip(detection_map, -1.0, 1.0)

        # delta mapping (post-processing)
        delta_val = float(modelConfig.get("delta", 0.012))
        detection_map = np.exp(-((detection_map - 1.0) ** 2) / delta_val)

        detection_map = np.reshape(detection_map, [h, w], order="F")
        detection_map = standard(detection_map)
        detection_map = np.clip(detection_map, 0, 1)

        # optional stats
        if verbose:
            background_data = detection_map[map == 0]
            target_data = detection_map[map == 1]

            print("\n背景数据统计信息：")
            print(f"  数据量: {len(background_data)}")
            print(f"  均值: {np.mean(background_data):.4f}")
            print(f"  标准差: {np.std(background_data):.4f}")
            print(f"  最小值: {np.min(background_data):.4f}")
            print(f"  最大值: {np.max(background_data):.4f}")

            print("\n目标数据统计信息：")
            print(f"  数据量: {len(target_data)}")
            print(f"  均值: {np.mean(target_data):.4f}")
            print(f"  标准差: {np.std(target_data):.4f}")
            print(f"  最小值: {np.min(target_data):.4f}")
            print(f"  最大值: {np.max(target_data):.4f}")

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # GPU memory (MB)
        if device.type == "cuda" and torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated(device) / 1024 ** 2
        else:
            memory_usage = 0.0

        y_l = np.reshape(map, [-1, 1], order="F")
        y_p = np.reshape(detection_map, [-1, 1], order="F")

        # calculate AUC1~AUC5
        fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        fpr = fpr[1:]
        tpr = tpr[1:]
        threshold = threshold[1:]

        auc1 = round(metrics.auc(fpr, tpr), modelConfig["epision"])
        auc2 = round(metrics.auc(threshold, fpr), modelConfig["epision"])
        auc3 = round(metrics.auc(threshold, tpr), modelConfig["epision"])
        auc4 = round(auc1 + auc3 - auc2, modelConfig["epision"])
        auc5 = round(auc3 / auc2, modelConfig["epision"]) if auc2 != 0 else 0

        print("\n=== Evaluation Metrics ===")
        print(f"Execution time: {execution_time:.4f} seconds")
        print(f"delta: {delta_val}")
        print(f'AUC1 (ROC AUC): {auc1:.{modelConfig["epision"]}f}')
        print(f'AUC2 (Threshold vs FPR): {auc2:.{modelConfig["epision"]}f}')
        print(f'AUC3 (Threshold vs TPR): {auc3:.{modelConfig["epision"]}f}')
        print(f'AUC4 (AUC1 + AUC3 - AUC2): {auc4:.{modelConfig["epision"]}f}')
        print(f'AUC5 (AUC3 / AUC2): {auc5:.{modelConfig["epision"]}f}')

        # -------- save outputs (optional) --------
        if save_visuals:
            gray_fname = os.path.join(result_dir, f"{Name}.png")
            fig1, ax1 = plt.subplots()
            ax1.imshow(detection_map, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax1.axis("off")
            fig1.savefig(gray_fname, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig1)

            color_fname = os.path.join(result_dir, f"{Name}_detMap_color.png")
            fig2, ax2 = plt.subplots()
            ax2.imshow(detection_map, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax2.axis("off")
            fig2.savefig(color_fname, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig2)

        if save_curves:
            if len(threshold) > 1:
                roc_3d_data = np.column_stack((threshold, fpr, tpr))
                np.savetxt(
                    os.path.join(result_dir, f"{Name}roc_3d.csv"),
                    roc_3d_data,
                    delimiter=",",
                    header="tau,pf,pd",
                    comments="",
                )

                roc_pf_pd_data = np.column_stack((fpr, tpr))
                np.savetxt(
                    os.path.join(result_dir, f"{Name}roc_pf_pd.csv"),
                    roc_pf_pd_data,
                    delimiter=",",
                    header="pf,pd",
                    comments="",
                )

                roc_tau_pf_data = np.column_stack((threshold, fpr))
                np.savetxt(
                    os.path.join(result_dir, f"{Name}roc_tau_pf.csv"),
                    roc_tau_pf_data,
                    delimiter=",",
                    header="tau,pf",
                    comments="",
                )

                roc_tau_pd_data = np.column_stack((threshold, tpr))
                np.savetxt(
                    os.path.join(result_dir, f"{Name}roc_tau_pd.csv"),
                    roc_tau_pd_data,
                    delimiter=",",
                    header="tau,pd",
                    comments="",
                )
            else:
                print("无法保存曲线数据：阈值数量不足")

    return {
        "AUC1": float(auc1),
        "AUC2": float(auc2),
        "AUC3": float(auc3),
        "AUC4": float(auc4),
        "AUC5": float(auc5),
        "execution_time_sec": float(execution_time),
        "memory_mb": float(memory_usage),
        "delta": float(delta_val),
    }
