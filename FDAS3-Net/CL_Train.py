# CL_Train.py (rev5)
# =============================================================================
# 主要改动（不改模型结构，仅增强“可调参性/稳定性/可视化一致性”）：
# 1) EnhancedBANLoss 里的 lambda_ban / lambda_null_pos / ban_beta / ban_m_pos / ban_m_neg
#    之前没有真正用起来（写在 cfg 里调参不生效）——已修复，且默认值保持与原实现一致。
# 2) 训练端增加 AMP dtype 选择与 NaN/Inf 守卫：
#    - cfg["amp"]=True 时启用 autocast
#    - cfg["amp_dtype"]="fp16" 或 "bf16"（bf16 下默认不启用 GradScaler）
#    - loss 非有限时跳过该步，避免把 NaN 写进权重
# 3) select_best / eval 的 target spectrum(ts) 生成与 Data_Patch 保持一致：
#    - 支持 cfg["ts_type"] 或 cfg["ts_ensemble"]（多策略集成）
# 4) eval 可视化：dsc_then_nbs(det_map, cfg=cfg) 允许按数据集调参；
#    并额外保存 raw det_map，便于判断“模型真的坏了”还是“显示拉伸坏了”。
#
# 备注：
# - 模型结构（CL_Model.py）不动
# - AUC 口径仍然是 HTD‑Mamba pipeline（AUC1..AUC5）
# =============================================================================

import os, time, glob, hashlib
from typing import Dict, Tuple, Optional, List

import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from sklearn import metrics

from CL_Model import SpectralGroupAttention
from Data_Patch import Data
from ts_generation import ts_generation
from Scheduler import GradualWarmupScheduler
from Utils import checkFile, standard, dsc_then_nbs, summarize_scores


# ----------------- Compatibility helpers ---------------- #
def _model_arch_signature(model: nn.Module) -> str:
    keys = sorted(list(model.state_dict().keys()))
    sig = hashlib.md5(("|".join(keys)).encode("utf-8")).hexdigest()[:8]
    return sig


def _ckpt_is_compatible(state_dict: Dict, expected_keys: set) -> bool:
    return set(state_dict.keys()) == expected_keys


# ----------------- Target spectrum helpers ---------------- #
# ----------------- Target spectrum helpers ---------------- #
def _ts_from_gtmask(data: np.ndarray, gt_mask: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    根据给定的 gt_mask（0/1）生成 target spectrum (B,1)。
    兼容：
      - cfg['ts_ensemble'] = [..]  多策略求平均
      - cfg['ts_type'] = int      单策略
    """
    ts_cfg = cfg.get("ts_ensemble", None)
    if isinstance(ts_cfg, (list, tuple)) and len(ts_cfg) > 0:
        ts_list = []
        for t in ts_cfg:
            try:
                ts_list.append(ts_generation(data, gt_mask, int(t)))
            except Exception:
                continue
        if len(ts_list) == 0:
            return ts_generation(data, gt_mask, int(cfg.get("ts_type", 7)))
        return np.mean(np.stack(ts_list, axis=0), axis=0)
    return ts_generation(data, gt_mask, int(cfg.get("ts_type", 7)))


def _sample_target_coords(gt: np.ndarray, k: int, seed: int) -> np.ndarray:
    """从 gt==1 的坐标里随机采样 k 个（可复现）。返回 shape=[k,2] 的 (row,col)。"""
    coords = np.argwhere(gt == 1)
    if coords.size == 0:
        raise ValueError("[ts] gt 中没有目标像元 (gt==1)，无法生成 target spectrum。")
    k = int(k)
    if k <= 0 or k >= coords.shape[0]:
        return coords
    rng = np.random.RandomState(int(seed))
    sel = rng.choice(coords.shape[0], size=k, replace=False)
    return coords[sel]


def _compute_target_spectrum(
    data: np.ndarray,
    gt: np.ndarray,
    cfg: Dict,
    *,
    return_selection: bool = False,
) -> np.ndarray:
    """
    统一的 ts 生成入口（用于 select_best / eval）：

    cfg['ts_mode'] 可选：
      - 'oracle' : 使用全部 gt==1 目标像元（你原始实现，指标可能非常“饱和”）
      - 'kshot'  : 只用 K 个目标像元估计目标签名（更接近 few-shot/工程可获得先验）
      - 'holdout': 用目标像元的一部分（比例 cfg['ts_frac']）估计签名

    返回：
      - ts: (B,1)
      - 如果 return_selection=True 且 ts_mode 是 kshot/holdout，则额外返回采样坐标 coords（用于评估时可选择排除这部分像元，避免信息泄漏）
    """
    mode = str(cfg.get("ts_mode", "oracle")).lower().strip()

    # --- oracle：全目标像元估计 ---
    if mode in ("oracle", "all", "full"):
        ts = _ts_from_gtmask(data, gt, cfg)
        if return_selection:
            return ts, None
        return ts

    # --- k-shot：少量目标点估计 ---
    if mode in ("kshot", "fewshot", "few_shot"):
        k = int(cfg.get("ts_k", cfg.get("ts_kshot", 5)))
        seed = int(cfg.get("ts_seed", cfg.get("ts_kshot_seed", cfg.get("seed", 0))))
        coords = _sample_target_coords(gt, k=k, seed=seed)
        gt_mask = np.zeros_like(gt, dtype=np.uint8)
        gt_mask[coords[:, 0], coords[:, 1]] = 1
        ts = _ts_from_gtmask(data, gt_mask, cfg)
        if return_selection:
            return ts, coords
        return ts

    # --- holdout：按比例采样 ---
    if mode in ("holdout", "fraction", "frac"):
        frac = float(cfg.get("ts_frac", 0.3))
        frac = float(np.clip(frac, 0.0, 1.0))
        coords_all = np.argwhere(gt == 1)
        if coords_all.size == 0:
            ts = _ts_from_gtmask(data, gt, cfg)
            if return_selection:
                return ts, None
            return ts
        k = max(1, int(round(coords_all.shape[0] * frac)))
        seed = int(cfg.get("ts_seed", cfg.get("seed", 0)))
        rng = np.random.RandomState(seed)
        sel = rng.choice(
            coords_all.shape[0], size=min(k, coords_all.shape[0]), replace=False
        )
        coords = coords_all[sel]
        gt_mask = np.zeros_like(gt, dtype=np.uint8)
        gt_mask[coords[:, 0], coords[:, 1]] = 1
        ts = _ts_from_gtmask(data, gt_mask, cfg)
        if return_selection:
            return ts, coords
        return ts

    # fallback
    ts = _ts_from_gtmask(data, gt, cfg)
    if return_selection:
        return ts, None
    return ts


# ----------------- Prototype manager ---------------- #


class BackgroundPrototypeManager:
    def __init__(self, K=4, feat_dim=205, momentum=0.99):
        self.K = int(K)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.prototypes = None  # [K, feat_dim]

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor):
        """Online KMeans update using background pixels (labels==0)."""
        bg_mask = labels == 0
        if bg_mask.sum() < self.K:
            return
        bg_feats = feats[bg_mask].detach().cpu().numpy()

        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=self.K, n_init=1, random_state=0).fit(bg_feats)
            new_proto = torch.from_numpy(km.cluster_centers_).float()
            if self.prototypes is None:
                self.prototypes = new_proto
            else:
                self.prototypes = (
                    self.momentum * self.prototypes + (1 - self.momentum) * new_proto
                )
        except Exception:
            # 允许在缺少 sklearn 或 KMeans 出错时继续训练
            pass

    def get(self, device):
        if self.prototypes is None:
            return None
        return self.prototypes.to(device)


# ----------------- BAN Loss (fixed tunables) ---------------- #
class EnhancedBANLoss(nn.Module):
    """
    BAN + Null-space positive constraint.

    NOTE: rev5 修复：lambda_ban / lambda_null_pos / ban_beta / ban_m_pos / ban_m_neg
    之前在 forward 里被写死了，导致 cfg 调参不生效。
    """

    def __init__(
        self,
        bg_manager: BackgroundPrototypeManager,
        lambda_ban: float = 0.35,
        lambda_null_pos: float = 0.25,
        ban_beta: float = 0.75,
        ban_m_pos: float = 0.30,
        ban_m_neg: float = 0.08,
    ):
        super().__init__()
        self.bg_manager = bg_manager
        self.lambda_ban = float(lambda_ban)
        self.lambda_null_pos = float(lambda_null_pos)
        self.ban_beta = float(ban_beta)
        self.ban_m_pos = float(ban_m_pos)
        self.ban_m_neg = float(ban_m_neg)

    def forward(self, feat0, feat1, target_feat, labels):
        """
        feat0/feat1: [B, D] (already normalized)
        target_feat: [1, D] / [D] / [1,1,D]（你工程里常见）/ 其它可广播形状
        labels: [B]
        """
        device = feat0.device

        # ---------------------------------------------------------------
        # 数值稳健性：BAN/QR 分解在 fp16 下很容易出问题（甚至直接不支持）。
        # 这里统一把参与 BAN 的张量转为 fp32 计算，避免 amp 引入的 NaN。
        # ---------------------------------------------------------------
        feat0 = feat0.float()
        feat1 = feat1.float()
        labels = labels.long()

        bg_protos = self.bg_manager.get(device)
        if bg_protos is None:
            z = torch.tensor(0.0, device=device)
            return {"total": z, "ban_loss": z, "null_pos_loss": z}

        # normalize
        bg_protos = F.normalize(bg_protos.float(), dim=1)

        # --- target_feat 形状兼容（你原工程中常见的是 [1,1,B]） ---
        tgt = target_feat
        if not torch.is_tensor(tgt):
            tgt = torch.as_tensor(tgt, device=device)
        tgt = tgt.to(device).float()

        # 允许: [D] / [1,D] / [1,1,D] / [K,1,D] / [B,K,D] ...
        if tgt.dim() == 0:
            tgt = tgt.view(1, 1)
        elif tgt.dim() == 1:
            tgt = tgt.view(1, -1)
        else:
            if tgt.dim() > 2:
                tgt = tgt.reshape(-1, tgt.shape[-1])  # flatten leading dims
            # 若有多个 target（例如集成/多原型），取均值作为单一 target
            if tgt.size(0) > 1:
                tgt = tgt.mean(dim=0, keepdim=True)

        tgt = F.normalize(tgt, dim=1)

        # null-space of background prototypes
        # Q: [D, K] (orthonormal basis)
        Q, _ = torch.linalg.qr(bg_protos.T)  # robust in fp32
        proj = Q @ Q.T  # [D, D]
        I = torch.eye(proj.shape[0], device=device, dtype=proj.dtype)
        null_proj = I - proj

        # similarity to target
        sim0_t = torch.mm(feat0, tgt.t()).squeeze(1)
        sim1_t = torch.mm(feat1, tgt.t()).squeeze(1)

        # similarity to background prototypes (take max)
        bg_sim0 = torch.mm(feat0, bg_protos.t()).max(dim=1).values
        bg_sim1 = torch.mm(feat1, bg_protos.t()).max(dim=1).values

        # BAN score: target-sim minus beta * bg-sim
        s0 = sim0_t - self.ban_beta * bg_sim0
        s1 = sim1_t - self.ban_beta * bg_sim1

        pos_mask = (labels == 1).float()
        neg_mask = (labels == 0).float()

        # margin losses
        loss_pos0 = F.relu(self.ban_m_pos - s0) * pos_mask
        loss_pos1 = F.relu(self.ban_m_pos - s1) * pos_mask
        loss_neg0 = F.relu(s0 + self.ban_m_neg) * neg_mask
        loss_neg1 = F.relu(s1 + self.ban_m_neg) * neg_mask

        denom = pos_mask.sum() + neg_mask.sum() + 1e-6
        ban_raw = (
            loss_pos0.sum() + loss_pos1.sum() + loss_neg0.sum() + loss_neg1.sum()
        ) / (denom * 4.0)
        ban_loss = self.lambda_ban * ban_raw

        # null-space constraint on positive samples
        # projected feature should have low similarity to background prototypes
        feat0_null = torch.matmul(feat0, null_proj)
        feat1_null = torch.matmul(feat1, null_proj)
        pos_bg_sim0 = torch.mm(feat0_null, bg_protos.t()).max(dim=1).values
        pos_bg_sim1 = torch.mm(feat1_null, bg_protos.t()).max(dim=1).values
        pos_cnt = pos_mask.sum() + 1e-6
        null_raw = ((pos_bg_sim0 * pos_mask).sum() + (pos_bg_sim1 * pos_mask).sum()) / (
            2.0 * pos_cnt
        )
        null_pos_loss = self.lambda_null_pos * null_raw

        total = ban_loss + null_pos_loss

        return {"total": total, "ban_loss": ban_loss, "null_pos_loss": null_pos_loss}


# ---------------- Balanced sampler ---------------- #
class BalanceBatchSampler(Sampler[List[int]]):
    """
    从全图像元中构造“正/负样本比例可控”的 batch。
    注意：这里的 label 来自 GT（你当前 pipeline 属于 target detection 的有监督设置）。
    """

    def __init__(
        self,
        dataset: Data,
        batch_size: int,
        pos_ratio: float = 0.25,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.pos_ratio = float(pos_ratio)
        self.shuffle = bool(shuffle)

        # ------------------------------------------------------------------
        # 兼容性处理：
        # 你原工程里的 Data 类并没有 reshape_order 属性（报错的根源）。
        # 但 Data.__getitem__ 的线性索引顺序由 index_order("F"/"C") 决定。
        # 因此这里应当按照同样的顺序把 labels 展平，保证 sampler 选出的 idx
        # 能正确对应到 (row, col)。
        # ------------------------------------------------------------------
        order = getattr(dataset, "reshape_order", None)
        if order is None:
            order = getattr(dataset, "index_order", None)
        if order is None:
            order = "F"  # 与 Data_Patch 默认一致
        order = str(order).upper()

        labels = dataset.labels
        # labels 可能是 torch tensor 或 numpy array，这里统一转 numpy
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()

        try:
            labels = labels.reshape(-1, order=order)
        except TypeError:
            # torch/numpy 某些版本 reshape 不支持 order 参数时回退
            labels = labels.reshape(-1)

        self.pos_idx = np.where(labels == 1)[0]
        self.neg_idx = np.where(labels == 0)[0]
        self.pos_per = max(1, int(self.batch_size * self.pos_ratio))
        self.neg_per = self.batch_size - self.pos_per
        self.num_batches = int(np.ceil(len(labels) / self.batch_size))
        self.rng = np.random.default_rng(12345)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        # 采用 “可重复但每个 epoch 都会变化” 的随机采样方式；
        # 并且在正/负样本不足时自动启用“有放回采样”，避免 batch 尺寸变动。
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            # 极端情况：label 全是 0 或全是 1（理论上不该发生，但防御一下）
            all_idx = np.arange(len(self.dataset))
            for _ in range(self.num_batches):
                batch = self.rng.choice(
                    all_idx,
                    size=self.batch_size,
                    replace=len(all_idx) < self.batch_size,
                )
                yield batch.tolist()
            return

        for _ in range(self.num_batches):
            pos_batch = self.rng.choice(
                self.pos_idx,
                size=self.pos_per,
                replace=len(self.pos_idx) < self.pos_per,
            )
            neg_batch = self.rng.choice(
                self.neg_idx,
                size=self.neg_per,
                replace=len(self.neg_idx) < self.neg_per,
            )
            batch = np.concatenate([pos_batch, neg_batch], axis=0)
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch.tolist()


# ---------------- EMA ---------------- #
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


# ---------------- Train ---------------- #
def train(cfg: Dict):
    device = cfg.get("device", "cuda:0")
    torch.manual_seed(cfg.get("seed", 1))
    np.random.seed(cfg.get("seed", 1))

    train_data = Data(cfg["path"], cfg["patch_size"], is_train=True, config=cfg)

    # ★ 按需启用“平衡Batch采样”
    if cfg.get("balance_sampler", False):
        sampler = BalanceBatchSampler(
            train_data,
            batch_size=cfg["batch_size"],
            pos_ratio=cfg.get("pos_ratio_in_batch", 0.25),
            shuffle=True,
        )
        train_loader = DataLoader(
            train_data, batch_sampler=sampler, num_workers=4, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    model = SpectralGroupAttention(
        band=cfg["band"],
        group_length=cfg["m"],
        channel_dim=cfg["channel"],
        state_size=cfg["state_size"],
        layer=cfg["layer"],
        use_wavelet=cfg.get("use_wavelet", True),
        wavelet_level=cfg.get("wavelet_level", 2),
        learnable_lp=cfg.get("learnable_lp", False),
        wavelet_name=cfg.get("wavelet_name", "db4"),
        attn_detach=cfg.get("attn_detach", True),
        multi_m=cfg.get("multi_m", [10, 14, 20]),
        m_gate_type=cfg.get("m_gate_type", "hilo"),
        use_bg_suppress=cfg.get("use_bg_suppress", True),
    ).to(device)

    # 让模型在前向时可感知目标谱（TASA）
    model.target = train_data.target.to(device)

    # 保存目录与签名
    save_root = os.path.join(cfg["save_dir"], cfg["dataset"])
    checkFile(save_root)
    arch_sig = _model_arch_signature(model)
    print(f"[Train] Arch signature = {arch_sig}")

    # 背景原型 & BANLoss
    bg_manager = BackgroundPrototypeManager(
        K=cfg.get("train_bg_K", 6),
        feat_dim=cfg["band"],
        momentum=cfg.get("bg_proto_momentum", 0.985),
    )
    ban_loss = EnhancedBANLoss(
        bg_manager=bg_manager,
        lambda_ban=cfg.get("lambda_ban", 0.35),
        lambda_null_pos=cfg.get("lambda_null_pos", 0.25),
        ban_beta=cfg.get("ban_beta", 0.75),
        ban_m_pos=cfg.get("ban_m_pos", 0.30),
        ban_m_neg=cfg.get("ban_m_neg", 0.08),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg["epoch"] - 5)
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=cfg.get("multiplier", 1.5),
        warm_epoch=5,
        after_scheduler=scheduler_cosine,
    )

    # AMP/EMA
    use_amp = bool(cfg.get("amp", False) and str(device).startswith("cuda"))
    amp_dtype_str = str(cfg.get("amp_dtype", "fp16")).lower()
    amp_dtype = (
        torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
    )

    # bf16 一般不需要 GradScaler；fp16 建议启用
    scaler_enabled = bool(use_amp and (amp_dtype == torch.float16))
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    use_ema = bool(cfg.get("use_ema", False))
    ema = EMA(model, decay=cfg.get("ema_decay", 0.998)) if use_ema else None

    # 动态困难负样本配置
    topk_start = float(cfg.get("hard_neg_topk_start", 0.05))
    topk_end = float(cfg.get("hard_neg_topk_end", 0.15))
    warm_epochs = int(cfg.get("hard_neg_warmup_epochs", 20))
    hard_margin = float(cfg.get("hard_neg_margin", 0.08))
    hard_lambda = float(cfg.get("lambda_hard_neg", 0.05))

    max_nan_steps = int(cfg.get("max_nan_steps", 20))
    nan_steps = 0

    for epoch in range(cfg["epoch"]):
        model.train()
        total_loss = 0.0
        with tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{cfg['epoch']}",
        ) as tbar:
            for it, (x0, x1, labels) in tbar:
                x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)

                # autocast（fp16/bf16）
                if use_amp:
                    try:
                        autocast_ctx = torch.cuda.amp.autocast(
                            enabled=True, dtype=amp_dtype
                        )
                    except TypeError:
                        autocast_ctx = torch.cuda.amp.autocast(enabled=True)
                else:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=False)

                with autocast_ctx:
                    feat0 = F.normalize(model(x0).squeeze(1), dim=1)
                    feat1 = F.normalize(model(x1).squeeze(1), dim=1)

                    # 更新背景原型（建议用 fp32）
                    if it % 10 == 0:
                        bg_manager.update(feat0.detach().float(), labels)

                    out = ban_loss(feat0, feat1, model.target, labels)
                    loss = out["total"]

                    # —— 动态困难负样本（Top‑k 比例随 epoch 线性上升）——
                    ratio = min(1.0, max(0.0, epoch / max(1, warm_epochs)))
                    topk_ratio = topk_start + (topk_end - topk_start) * ratio
                    hard_neg_loss = torch.tensor(0.0, device=device)

                    tgt = F.normalize(model.target.squeeze().view(1, -1), dim=1)
                    sim0 = torch.mm(feat0, tgt.t()).squeeze(1)
                    sim1 = torch.mm(feat1, tgt.t()).squeeze(1)
                    neg_mask = labels == 0
                    if neg_mask.any() and cfg.get("use_hard_neg", True):
                        sim_neg = torch.cat([sim0[neg_mask], sim1[neg_mask]], dim=0)
                        k = max(1, int(sim_neg.numel() * topk_ratio))
                        topk_vals, _ = torch.topk(sim_neg, k)
                        hard_neg_loss = (
                            F.relu(topk_vals - hard_margin).mean() * hard_lambda
                        )
                        loss = loss + hard_neg_loss

                # --- NaN/Inf 守卫：避免把 NaN 写进权重 ---
                if not torch.isfinite(loss):
                    nan_steps += 1
                    print(
                        f"[WARN] Non-finite loss at Ep {epoch+1} It {it+1}: {loss.detach().cpu().item()}  (skip step, nan_steps={nan_steps})"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if scaler_enabled:
                        # GradScaler.update() 正常应在 scaler.step(optimizer) 之后调用。
                        # 这里因为 loss 非有限我们跳过 backward/step，所以必须传 new_scale
                        # 来避免 PyTorch 断言，同时也能清空 scaler 的 per-optimizer 状态。
                        cur_scale = float(scaler.get_scale())
                        backoff = float(
                            cfg.get("amp_backoff_factor", 0.5)
                        )  # 可选：遇到 NaN/Inf 时回退
                        scaler.update(max(cur_scale * backoff, 1.0))
                    if nan_steps >= max_nan_steps:
                        print("[WARN] Too many non-finite steps, stop training early.")
                        break
                    continue

                # --- backward & step ---
                optimizer.zero_grad(set_to_none=True)

                if scaler_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.get("grad_clip", 0.5)
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # fp32 / bf16
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.get("grad_clip", 0.5)
                    )
                    optimizer.step()

                if ema is not None:
                    ema.update(model)

                total_loss += float(loss.item())

                if (it + 1) % int(cfg.get("log_interval", 50)) == 0:
                    dbg = getattr(model, "last_debug", {})
                    hi_lo = dbg.get("hi_lo_ratio", None)
                    try:
                        hi_lo_str = f"{float(hi_lo):.3f}" if hi_lo is not None else "NA"
                    except Exception:
                        hi_lo_str = "NA"
                    print(
                        f"[Train][Ep {epoch+1} | It {it+1}] "
                        f"loss={float(loss.item()):.4f} (ban={float(out['ban_loss'].item()):.4f} null_pos={float(out['null_pos_loss'].item()):.4f} hard_neg={float(hard_neg_loss.item()):.4f}) "
                        f"bg_prob={dbg.get('bg_prob_mean','NA')}  hi/lo={hi_lo_str} "
                        f"gate_w={dbg.get('gate_w_mean','NA')}  Top5Bands={dbg.get('att_top5_idx', [])}  "
                        f"topk_ratio={topk_ratio:.3f}"
                    )

                tbar.set_postfix(
                    ordered_dict={
                        "loss": f"{float(loss.item()):.4f}",
                        "LR": optimizer.param_groups[0]["lr"],
                    }
                )

            # tqdm loop end

        scheduler.step()

        # —— 每个 epoch 保存 ckpt —— #
        ckpt_path = os.path.join(save_root, f"ckpt_{epoch}_.pt")
        if ema is not None:
            torch.save(ema.state_dict(), ckpt_path)  # 保存 EMA 权重
        else:
            torch.save(model.state_dict(), ckpt_path)
        print(f"[Train] Saved checkpoint -> {ckpt_path}")

        if nan_steps >= max_nan_steps:
            break

    print("[Train] 完成。")


# ---------------- HTD‑style select_best ---------------- #
def select_best(cfg: Dict):
    """
    选择最佳权重（HTD‑Mamba 口径）：

    重要改动（为了解决你现在“只想 SelectBest+Eval，不想再训练”的需求，并避免显存问题）：
      1) 不再依赖 cfg["epoch"] 的范围，而是直接扫描 models/<dataset>/ckpt_*_.pt
         —— 这样即使训练中断/epoch 写错，也能正常选最优。
      2) 评估 batch 使用 cfg["eval_batch_size"]（若未提供则退回 cfg["batch_size"]）
         —— 避免 eval/select_best 用训练 batch 导致 OOM。
      3) 遇到 CUDA OOM 自动把 eval_batch_size 砍半重试（直到 1）
         —— 尽量做到“不要报错，至少能跑完”。
      4) 默认每个 ckpt 后 empty_cache（可通过 cfg["empty_cache_each_ckpt"]=False 关闭）
    """
    device = torch.device(cfg["device"])
    path = os.path.join(cfg["save_dir"], cfg["dataset"])
    checkFile(path)

    mat = sio.loadmat(cfg["path"])
    data = standard(mat["data"])
    gt = mat["map"]
    ts = _compute_target_spectrum(data, gt, cfg)

    h, w, c = data.shape
    numpixel = h * w
    data_mat = np.reshape(data, [-1, c], order=cfg.get("reshape_order", "F"))

    # ---- 构造模型（只构一次，循环加载不同 ckpt）----
    model = SpectralGroupAttention(
        band=c,
        group_length=cfg["m"],
        channel_dim=cfg["channel"],
        state_size=cfg["state_size"],
        layer=cfg["layer"],
        use_wavelet=cfg.get("use_wavelet", True),
        wavelet_level=cfg.get("wavelet_level", 2),
        learnable_lp=cfg.get("learnable_lp", False),
        wavelet_name=cfg.get("wavelet_name", "db4"),
        attn_detach=cfg.get("attn_detach", True),
        multi_m=cfg.get("multi_m", [10, 14, 20]),
        m_gate_type=cfg.get("m_gate_type", "hilo"),
        use_bg_suppress=cfg.get("use_bg_suppress", True),
    ).to(device)
    expected_keys = set(model.state_dict().keys())

    # ---- 扫描 ckpt 文件 ----
    ckpt_paths = sorted(glob.glob(os.path.join(path, "ckpt_*_.pt")))
    if len(ckpt_paths) == 0:
        raise RuntimeError(f"[select_best] 未找到 ckpt 文件：{path}/ckpt_*_.pt")

    def _parse_epoch(p: str) -> int:
        base = os.path.basename(p)
        # ckpt_{e}_.pt
        try:
            mid = base.replace("ckpt_", "").replace("_.pt", "")
            return int(mid)
        except Exception:
            return -1

    ckpts = [(e, p) for p in ckpt_paths for e in [_parse_epoch(p)] if e >= 0]
    ckpts.sort(key=lambda x: x[0])

    # ---- 选择指标：auc1 / auc4 / auc5 ----
    select_metric = str(cfg.get("select_metric", "auc1")).lower()

    # ---- 评估 batch：用 eval_batch_size（不跟训练 batch 绑定）----
    eval_batch_default = int(cfg.get("eval_batch_size", cfg.get("batch_size", 64)))
    eval_batch_default = max(1, eval_batch_default)

    # eval 端可选 AMP（默认关闭）
    use_amp = bool(cfg.get("eval_amp", False) and device.type == "cuda")
    amp_dtype_str = str(cfg.get("eval_amp_dtype", cfg.get("amp_dtype", "fp16"))).lower()
    amp_dtype = (
        torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
    )

    empty_cache_each_ckpt = bool(cfg.get("empty_cache_each_ckpt", True))

    max_score, opt_epoch = -1.0, -1
    found_compatible = False

    # 记录每轮指标（用户要求：select 每一轮都打印 auc1~auc5）
    # 字段：epoch, auc1..auc5, score(select_metric), used_bs
    sel_rows = []

    # 是否逐轮打印 AUC1~AUC5（默认 True）
    print_each = bool(cfg.get("select_print_each", True))
    # 是否保存一份 select 过程的 CSV（默认 True）
    save_csv = bool(cfg.get("select_save_csv", True))

    for e, ck in ckpts:
        # --- 兼容性检查 ---
        raw_state = torch.load(ck, map_location="cpu")
        if not _ckpt_is_compatible(raw_state, expected_keys):
            print(
                f"[select_best] Skip incompatible ckpt: {os.path.basename(ck)} (state_dict keys mismatch)"
            )
            continue
        found_compatible = True

        # --- load ---
        model.load_state_dict(raw_state, strict=True)
        model.target = torch.from_numpy(ts.T).float().unsqueeze(0).to(device)  # [1,1,B]
        model.eval()

        # --- target feature ---
        with torch.no_grad():
            if use_amp:
                try:
                    autocast_ctx = torch.cuda.amp.autocast(
                        enabled=True, dtype=amp_dtype
                    )
                except TypeError:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=False)

            with autocast_ctx:
                t_prior = (
                    torch.from_numpy(ts.T).float().unsqueeze(1).to(device)
                )  # [1,1,B]
                t_feat_t = model(t_prior)  # [1,1,D] or [1,D]
            t_feat = t_feat_t.detach().float().cpu().numpy().reshape(1, -1)

            # --- full-pixel inference (OOM safe) ---
            det = np.zeros([numpixel], dtype=np.float32)
            bs = int(eval_batch_default)
            i = 0
            while i < numpixel:
                j = min(numpixel, i + bs)
                px = torch.from_numpy(data_mat[i:j]).to(device).unsqueeze(1)  # [bs,1,B]
                try:
                    with autocast_ctx:
                        feat_t = model(px)
                    feat = feat_t.detach().float().cpu().numpy().reshape(px.size(0), -1)

                    x_norm = np.sqrt(np.sum(feat**2, axis=1)) + 1e-8
                    y_norm = np.sqrt(np.sum(t_feat**2, axis=1)) + 1e-8
                    det[i:j] = (
                        np.sum(feat * t_feat, axis=1) / (x_norm * y_norm)
                    ).astype(np.float32)

                    i = j  # advance
                except RuntimeError as ex:
                    msg = str(ex).lower()
                    if ("out of memory" in msg) and (device.type == "cuda"):
                        # OOM: reduce batch and retry
                        try:
                            del px, feat_t
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                        new_bs = max(1, bs // 2)
                        if new_bs == bs:
                            raise
                        bs = new_bs
                        print(
                            f"[select_best] CUDA OOM -> reduce eval_batch_size to {bs} and retry ..."
                        )
                        continue
                    raise

            # --- det -> det_map ---
            delta = float(cfg.get("delta", 0.08))
            det_map = np.exp(-1.0 * (det - 1.0) ** 2 / delta)
            det_map = np.reshape(det_map, [h, w], order=cfg.get("reshape_order", "F"))
            det_map = standard(det_map)
            det_map = np.clip(det_map, 0, 1)

            if not np.isfinite(det_map).all():
                print(f"[select_best] Non-finite det_map at epoch {e}, skip.")
                continue

            # --- metrics ---
            y_l = np.reshape(gt, [-1, 1], order=cfg.get("reshape_order", "F"))
            y_p = np.reshape(det_map, [-1, 1], order=cfg.get("reshape_order", "F"))
            try:
                fpr, tpr, thr = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
                fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]
                auc1 = float(metrics.auc(fpr, tpr)) if len(fpr) > 1 else 0.5
                auc2 = float(metrics.auc(thr, fpr)) if len(thr) > 1 else 0.0
                auc3 = float(metrics.auc(thr, tpr)) if len(thr) > 1 else 0.0
                auc4 = auc1 + auc3 - auc2 if len(thr) > 1 else 0.0
                auc5 = (auc3 / auc2) if (auc2 is not None and auc2 != 0) else 0.0
            except Exception:
                auc1, auc2, auc3, auc4, auc5 = 0.5, 0.0, 0.0, 0.0, 0.0

            if select_metric == "auc5":
                score = float(auc5)
            elif select_metric == "auc4":
                score = float(auc4)
            elif select_metric == "auc3":
                score = float(auc3)
            else:
                score = float(auc1)

            # --- 逐轮打印/记录 ---
            sel_rows.append((e, auc1, auc2, auc3, auc4, auc5, score, int(bs)))
            if print_each:
                ep = int(cfg.get("epision", 4))
                print(
                    f"[select_best] epoch={e:03d} | "
                    f"AUC1={auc1:.{ep}f}  AUC2={auc2:.{ep}f}  AUC3={auc3:.{ep}f}  AUC4={auc4:.{ep}f}  AUC5={auc5:.{ep}f}  "
                    f"| score({select_metric})={score:.{ep}f} | bs={bs}"
                )

        if score > max_score:
            max_score, opt_epoch = score, e

        if empty_cache_each_ckpt and device.type == "cuda":
            torch.cuda.empty_cache()

    if not found_compatible:
        raise RuntimeError(
            "[select_best] 没有找到与当前模型架构兼容的 ckpt；请先完成一次训练或清理旧权重后重试。"
        )

    # 保存 select 全过程 CSV，便于你画每轮 AUC 曲线
    if save_csv and len(sel_rows) > 0:
        try:
            out_dir = cfg.get("select_out_dir", "result")
            checkFile(out_dir)
            out_path = os.path.join(
                out_dir, f"{cfg.get('dataset','dataset')}_select_auc.csv"
            )
            arr = np.array(sel_rows, dtype=np.float64)
            header = "epoch,auc1,auc2,auc3,auc4,auc5,score,eval_bs"
            np.savetxt(out_path, arr, delimiter=",", header=header, comments="")
            # 同时打印一下 Top-5，方便快速定位
            sel_sorted = sorted(sel_rows, key=lambda x: x[6], reverse=True)
            print("[select_best] Top-5 by", select_metric)
            for r in sel_sorted[:5]:
                e, a1, a2, a3, a4, a5, sc, bs_used = r
                print(
                    f"  epoch={int(e):03d}  score={sc:.4f}  (AUC1={a1:.4f} AUC2={a2:.4f} AUC3={a3:.4f} AUC4={a4:.4f} AUC5={a5:.4f})"
                )
            print(f"[select_best] per-epoch metrics saved -> {out_path}")
        except Exception as ex:
            print(f"[select_best] WARN: failed to save select CSV: {ex}")

    print(
        f"[select_best] 最佳 Epoch: {opt_epoch}  ({select_metric.upper()}={max_score:.{cfg['epision']}f})"
    )
    return round(max_score, cfg["epision"]), opt_epoch


# ---------------- HTD‑style eval ---------------- #
def eval(cfg: Dict):
    """
    最终评估（HTD‑Mamba 口径） + 更“论文友好”的输出：
      - det_map: 参与 ROC/AUC 的检测图（标准化到 [0,1]）
      - det_vis: 仅用于可视化（dsc_then_nbs / clip_pow 等），不影响 AUC
      - 额外输出：在指定 PF（虚警率）下的二值检测 mask + overlay（让 Sandiego 可视化更“像检测结果”）

    另外支持 cfg['eval_profiles']：
      你可以在不重训的情况下，用不同的 ts 估计方式（oracle/kshot）做多套评估，
      例如 abu-beach-4 的 oracle 指标可能会“饱和到不敢写”，而 k-shot 会更接近真实 few-shot 场景。
    """
    start = time.perf_counter()
    device = torch.device(cfg["device"])

    ckpt_dir = os.path.join(cfg["save_dir"], cfg["dataset"])
    result_dir = "result"
    checkFile(result_dir)

    # ---------- 一些小工具 ----------
    def _fmt_pf(pf: float) -> str:
        pf = float(pf)
        if pf <= 0:
            return "0"
        # 1e-3 -> "1e-3"
        exp = int(np.floor(np.log10(pf)))
        mant = pf / (10**exp)
        if abs(mant - 1.0) < 1e-6:
            return f"1e{exp}"
        return f"{pf:.3g}"

    def _pd_at_pf(fpr: np.ndarray, tpr: np.ndarray, pf: float) -> float:
        pf = float(pf)
        idx = np.where(fpr <= pf)[0]
        if idx.size == 0:
            return float(tpr[0]) if tpr.size > 0 else 0.0
        return float(tpr[idx[-1]])

    def _pauc_norm(fpr: np.ndarray, tpr: np.ndarray, pf_max: float) -> float:
        """归一化 pAUC：积分区间 [0,pf_max]，再除以 pf_max，范围 [0,1]。"""
        pf_max = float(pf_max)
        if pf_max <= 0:
            return 0.0
        # 确保包含 pf_max 点
        if fpr.size < 2:
            return 0.0
        pf_max = min(pf_max, float(np.max(fpr)))
        t_at = float(np.interp(pf_max, fpr, tpr))
        f2 = np.concatenate([fpr, np.array([pf_max], dtype=np.float32)])
        t2 = np.concatenate([tpr, np.array([t_at], dtype=np.float32)])
        m = f2 <= pf_max + 1e-12
        try:
            area = float(metrics.auc(f2[m], t2[m]))
        except Exception:
            area = 0.0
        return float(area / max(1e-12, pf_max))

    def _tau_for_pf(fpr: np.ndarray, thr: np.ndarray, pf: float) -> float:
        pf = float(pf)
        idx = np.where(fpr <= pf)[0]
        if idx.size == 0:
            return float(thr[0])
        return float(thr[idx[-1]])

    def _filter_small_cc(mask: np.ndarray, min_size: int = 1) -> np.ndarray:
        """兼容旧接口：仅按最小面积过滤连通域。"""
        return _filter_cc_by_size(mask, min_size=min_size, max_size=None)

    def _filter_cc_by_size(
        mask: np.ndarray, min_size: int = 1, max_size: Optional[int] = None
    ) -> np.ndarray:
        """按连通域面积过滤：保留 [min_size, max_size] 范围内的连通域。

        说明：
        - 这个只用于可视化（mask/heatmap 清理），不影响 AUC 计算。
        - max_size=None 表示不限制最大面积。
        """
        min_size = int(min_size)
        max_size = None if max_size is None else int(max_size)
        if min_size <= 1 and (max_size is None):
            return mask
        try:
            import scipy.ndimage as ndi

            lab, num = ndi.label(mask.astype(np.uint8))
            if num <= 0:
                return mask
            sizes = np.bincount(lab.ravel())
            keep = np.ones_like(sizes, dtype=bool)
            keep[0] = False
            if min_size > 1:
                keep &= sizes >= min_size
            if max_size is not None and max_size > 0:
                keep &= sizes <= max_size
            out = keep[lab]
            return out.astype(bool)
        except Exception:
            return mask

    def _tau_from_neg_pf(det_map: np.ndarray, gt_map: np.ndarray, pf: float) -> float:
        """用负样本(背景)分位数近似 PF 阈值：保证背景中大约只有 pf 比例会被判为正。

        比直接用 roc_curve 的 thr 更稳健，尤其是 det_map 极稀疏/有大量重复值时。
        返回 tau，并建议使用 score > tau（严格大于）来生成 mask。
        """
        pf = float(pf)
        neg = det_map[gt_map == 0].reshape(-1)
        if neg.size == 0:
            return 1.0
        neg = neg.astype(np.float64)
        n = int(neg.size)
        k = int(np.ceil(pf * n))
        if k <= 0:
            # 极低 PF：阈值略大于背景最大值
            tau = float(np.max(neg))
            return float(np.nextafter(tau, np.inf))
        # k-th largest => index n-k in ascending order
        idx = max(0, n - k)
        tau = float(np.partition(neg, idx)[idx])
        # 用 nextafter 提高一点点，配合严格 >，让实际 PF 不超过目标
        return float(np.nextafter(tau, np.inf))

    def _median_filter01(img01: np.ndarray, ksize: int = 3) -> np.ndarray:
        """对 [0,1] 图做中值滤波（仅用于可视化去散点）。"""
        ksize = int(ksize)
        if ksize <= 1:
            return img01
        try:
            import cv2

            if ksize % 2 == 0:
                ksize += 1
            u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
            out = cv2.medianBlur(u8, ksize)
            return out.astype(np.float32) / 255.0
        except Exception:
            return img01

    # ---------- 读取数据 ----------
    with torch.no_grad():
        mat = sio.loadmat(cfg["path"])
        data = standard(mat["data"])
        gt = mat["map"]

    h, w, c = data.shape
    numpixel = h * w
    reshape_order = cfg.get("reshape_order", "F")
    data_mat = np.reshape(data, [-1, c], order=reshape_order)

    # 作为 overlay 的背景（均值图），不影响指标
    bg_img = standard(np.mean(data, axis=2))

    # ---------- 构造模型 + 加载权重 ----------
    model = SpectralGroupAttention(
        band=c,
        group_length=cfg["m"],
        channel_dim=cfg["channel"],
        state_size=cfg["state_size"],
        layer=cfg["layer"],
        use_wavelet=cfg.get("use_wavelet", True),
        wavelet_level=cfg.get("wavelet_level", 2),
        learnable_lp=cfg.get("learnable_lp", False),
        wavelet_name=cfg.get("wavelet_name", "db4"),
        attn_detach=cfg.get("attn_detach", True),
        multi_m=cfg.get("multi_m", [10, 14, 20]),
        m_gate_type=cfg.get("m_gate_type", "hilo"),
        use_bg_suppress=cfg.get("use_bg_suppress", True),
    ).to(device)

    expected_keys = set(model.state_dict().keys())
    ckpt_path = os.path.join(ckpt_dir, cfg["test_load_weight"])
    raw_state = torch.load(ckpt_path, map_location=device)
    if not _ckpt_is_compatible(raw_state, expected_keys):
        raise RuntimeError(
            f"[eval] 权重与当前模型不兼容：{cfg['test_load_weight']}。请用 select_best 的最优 ckpt。"
        )
    model.load_state_dict(raw_state, strict=True)
    model.eval()

    # eval AMP（默认关闭）
    use_amp = bool(cfg.get("eval_amp", False) and device.type == "cuda")
    amp_dtype_str = str(cfg.get("eval_amp_dtype", cfg.get("amp_dtype", "fp16"))).lower()
    amp_dtype = (
        torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
    )

    if use_amp:
        try:
            autocast_ctx = torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)
        except TypeError:
            autocast_ctx = torch.cuda.amp.autocast(enabled=True)
    else:
        autocast_ctx = torch.cuda.amp.autocast(enabled=False)

    # 推理 batch（独立于训练 batch）
    eval_batch_default = int(cfg.get("eval_batch_size", cfg.get("batch_size", 64)))
    eval_batch_default = max(1, eval_batch_default)

    # ---------- profiles：允许多套评估（不重训） ----------
    profiles = cfg.get("eval_profiles", None)
    if not isinstance(profiles, (list, tuple)) or len(profiles) == 0:
        profiles = [dict(name="", ts_mode=str(cfg.get("ts_mode", "oracle")))]

    import matplotlib.pyplot as plt

    for pi, prof in enumerate(profiles):
        # profile 覆盖 cfg（不修改原 cfg）
        pcfg = dict(cfg)
        if isinstance(prof, dict):
            pcfg.update(prof)

        pname = str(pcfg.get("name", "")).strip()
        suffix = f"_{pname}" if pname else ""

        # ----- 生成 ts（可选 k-shot），并决定是否排除采样点 -----
        ts, sel_coords = _compute_target_spectrum(data, gt, pcfg, return_selection=True)

        exclude_from_eval = bool(pcfg.get("ts_exclude_from_eval", False))
        valid_mask = None
        if exclude_from_eval and (sel_coords is not None) and (len(sel_coords) > 0):
            gt_eval = gt.astype(np.int16).copy()
            gt_eval[sel_coords[:, 0], sel_coords[:, 1]] = -1
            y_l_full = np.reshape(gt_eval, [-1], order=reshape_order)
            valid_mask = y_l_full != -1
        else:
            y_l_full = np.reshape(gt, [-1], order=reshape_order)
            valid_mask = np.ones_like(y_l_full, dtype=bool)

        # ----- 设置 target prior（模型是 target-aware 的，所以 profile 之间必须重新跑 forward） -----
        model.target = torch.from_numpy(ts.T).float().unsqueeze(0).to(device)  # [1,1,B]

        # ----- 目标特征 -----
        with torch.no_grad():
            with autocast_ctx:
                t_prior = (
                    torch.from_numpy(ts.T).float().unsqueeze(1).to(device)
                )  # [1,1,B]
                t_feat_t = model(t_prior)
        t_feat = t_feat_t.detach().float().cpu().numpy().reshape(1, -1)

        # ----- 全像元推理（OOM safe） -----
        det = np.zeros([numpixel], dtype=np.float32)
        bs = int(eval_batch_default)
        i = 0
        while i < numpixel:
            j = min(numpixel, i + bs)
            px = torch.from_numpy(data_mat[i:j]).to(device).unsqueeze(1)  # [bs,1,B]
            try:
                with torch.no_grad():
                    with autocast_ctx:
                        feat_t = model(px)
                feat = feat_t.detach().float().cpu().numpy().reshape(px.size(0), -1)

                x_norm = np.sqrt(np.sum(feat**2, axis=1)) + 1e-8
                y_norm = np.sqrt(np.sum(t_feat**2, axis=1)) + 1e-8
                det[i:j] = (np.sum(feat * t_feat, axis=1) / (x_norm * y_norm)).astype(
                    np.float32
                )
                i = j
            except RuntimeError as ex:
                msg = str(ex).lower()
                if ("out of memory" in msg) and (device.type == "cuda"):
                    try:
                        del px, feat_t
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                    new_bs = max(1, bs // 2)
                    if new_bs == bs:
                        raise
                    bs = new_bs
                    print(
                        f"[eval{suffix}] CUDA OOM -> reduce eval_batch_size to {bs} and retry ..."
                    )
                    continue
                raise

        # ----- delta 网格选择（profile 可覆盖） -----
        delta_grid = pcfg.get("delta_grid", None)
        delta_select_metric = str(pcfg.get("delta_select_metric", "auc5")).lower()

        def _eval_one_delta(delta_val: float):
            d = float(delta_val)
            det_map = np.exp(-1.0 * (det - 1.0) ** 2 / d)
            det_map = np.reshape(det_map, [h, w], order=reshape_order)
            det_map = standard(det_map)
            det_map = np.clip(det_map, 0, 1)

            y_p_full = np.reshape(det_map, [-1], order=reshape_order)
            y_l = y_l_full[valid_mask].astype(np.int32)
            y_p = y_p_full[valid_mask].astype(np.float32)

            fpr, tpr, thr = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
            # HTD 口径：去掉第一个点（thr=inf 的哨兵点）
            fpr1, tpr1, thr1 = fpr[1:], tpr[1:], thr[1:]
            auc1 = float(metrics.auc(fpr1, tpr1)) if len(fpr1) > 1 else 0.5
            auc2 = float(metrics.auc(thr1, fpr1)) if len(thr1) > 1 else 0.0
            auc3 = float(metrics.auc(thr1, tpr1)) if len(thr1) > 1 else 0.0
            auc4 = auc1 + auc3 - auc2 if len(thr1) > 1 else 0.0
            auc5 = (auc3 / auc2) if (auc2 is not None and auc2 != 0) else 0.0
            return det_map, fpr, tpr, thr, auc1, auc2, auc3, auc4, auc5

        if isinstance(delta_grid, (list, tuple)) and len(delta_grid) > 0:
            best = None
            for d in delta_grid:
                det_map_d, fpr_d, tpr_d, thr_d, auc1, auc2, auc3, auc4, auc5 = (
                    _eval_one_delta(float(d))
                )
                if delta_select_metric == "auc1":
                    score = auc1
                elif delta_select_metric == "auc4":
                    score = auc4
                elif delta_select_metric == "auc3":
                    score = auc3
                else:
                    score = auc5
                if (best is None) or (score > best["score"]):
                    best = {
                        "delta": float(d),
                        "score": float(score),
                        "det_map": det_map_d,
                        "fpr": fpr_d,
                        "tpr": tpr_d,
                        "thr": thr_d,
                        "auc1": auc1,
                        "auc2": auc2,
                        "auc3": auc3,
                        "auc4": auc4,
                        "auc5": auc5,
                    }
            assert best is not None
            det_map = best["det_map"]
            fpr, tpr, thr = best["fpr"], best["tpr"], best["thr"]
            auc1, auc2, auc3, auc4, auc5 = (
                best["auc1"],
                best["auc2"],
                best["auc3"],
                best["auc4"],
                best["auc5"],
            )
            delta_used = best["delta"]
            print(
                f"[eval{suffix}] delta_grid enabled: best delta={delta_used} by {delta_select_metric} (score={best['score']:.6f})"
            )
        else:
            delta_used = float(pcfg.get("delta", 0.08))
            det_map, fpr, tpr, thr, auc1, auc2, auc3, auc4, auc5 = _eval_one_delta(
                delta_used
            )

        # ----- 额外：低虚警指标（更适合 Sandiego 的展示） -----
        pf_list = pcfg.get("viz_pf_list", pcfg.get("pf_list", [1e-3]))
        if not isinstance(pf_list, (list, tuple)):
            pf_list = [pf_list]

        pd_list = {float(pf): _pd_at_pf(fpr, tpr, float(pf)) for pf in pf_list}
        pauc_list = {float(pf): _pauc_norm(fpr, tpr, float(pf)) for pf in pf_list}

        # ----- 指标打印 -----
        ep = int(pcfg.get("epision", 4))
        info_ts = f"ts_mode={pcfg.get('ts_mode','oracle')}  ts_type={pcfg.get('ts_type',None)}"
        if isinstance(pcfg.get("ts_ensemble", None), (list, tuple)):
            info_ts = f"ts_mode={pcfg.get('ts_mode','oracle')}  ts_ensemble={pcfg.get('ts_ensemble')}"
        if sel_coords is not None:
            info_ts += f"  (selected={len(sel_coords)})"
            if exclude_from_eval:
                info_ts += "  [excluded from eval]"

        elapsed = time.perf_counter() - start
        print(
            "\n=== HTD‑style Evaluation Metrics"
            + (f" ({pname})" if pname else "")
            + " ==="
        )
        print(f"{info_ts} | delta={delta_used:.4f} | time={elapsed:.2f}s")
        print(
            f"AUC1={auc1:.{ep}f}  AUC2={auc2:.{ep}f}  AUC3={auc3:.{ep}f}  AUC4={auc4:.{ep}f}  AUC5={auc5:.{ep}f}"
        )
        for pf in pf_list:
            pf_f = float(pf)
            print(
                f"  PD@PF={pf_f:g} = {pd_list[pf_f]:.{ep}f}   pAUC_norm@PF={pf_f:g} = {pauc_list[pf_f]:.{ep}f}"
            )

        # ----- 可视化后处理（不影响 AUC） -----
        det_vis = dsc_then_nbs(
            det_map, cfg=pcfg, mode=pcfg.get("postproc_mode", "auto")
        )
        diag = summarize_scores(det_map, det_vis)
        print(f"[Viz{suffix}] diag: {diag}")

        Name = str(pcfg["dataset"])

        # --- 保存背景图（便于 overlay 对齐） ---
        bg_fname = os.path.join(result_dir, f"{Name}{suffix}_bg_gray.png")
        plt.figure()
        plt.imshow(bg_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        plt.axis("off")
        plt.savefig(bg_fname, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        # --- 保存 det_map 与 det_vis ---
        raw_gray_fname = os.path.join(result_dir, f"{Name}{suffix}_detMap_raw_gray.png")
        plt.figure()
        plt.imshow(det_map, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        plt.axis("off")
        plt.savefig(raw_gray_fname, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        gray_fname = os.path.join(result_dir, f"{Name}{suffix}_detMap_gray.png")
        plt.figure()
        plt.imshow(det_vis, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        plt.axis("off")
        plt.savefig(gray_fname, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        color_fname = os.path.join(result_dir, f"{Name}{suffix}_detMap_color.png")
        plt.figure()
        plt.imshow(det_vis, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        plt.axis("off")
        plt.savefig(color_fname, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        # ------------------------------------------------------------------
        # 额外输出：更“干净”的 viridis 热力图版本（不影响 AUC，仅用于论文可视化挑选）
        # 典型用途：Sandiego 指标很好，但 detMap 背景纹理/结构响应会让视觉看起来“很差”。
        # 解决：用 PF 约束阈值（基于负样本分位数）只保留最可信的 top 响应，再画 viridis。
        #
        # 通过在脚本里设置 pcfg['viridis_variants'] 来控制生成哪些版本。
        # 每个元素示例：
        #   {"tag":"pf1e-3", "mode":"pf_mask", "pf":1e-3, "min_cc":3, "max_cc":None, "rescale":True, "median":3}
        #   {"tag":"clip9995_g8", "mode":"clip_pow", "clip_p":99.95, "pow_gamma":8}
        # ------------------------------------------------------------------
        variants = pcfg.get("viridis_variants", None)
        if isinstance(variants, (list, tuple)) and len(variants) > 0:
            for v in variants:
                if not isinstance(v, dict):
                    continue
                tag = str(v.get("tag", "v"))
                mode = str(v.get("mode", "clip_pow")).lower().strip()

                # variant 独立 cfg（不污染 pcfg）
                vcfg = dict(pcfg)
                vcfg.update(v)

                try:
                    if mode in ("raw", "det_map"):
                        heat = det_map
                    elif mode in ("clip_pow", "pow", "rank"):
                        # 复用 dsc_then_nbs 的轻量模式（只做绝对/分位压缩）
                        heat = dsc_then_nbs(det_map, cfg=vcfg, mode=mode)
                    elif mode in ("pf", "pf_mask", "pf_masked"):
                        pf_v = float(vcfg.get("pf", 1e-3))
                        tau_q = _tau_from_neg_pf(det_map, gt, pf_v)
                        # 严格 > tau（更接近“PF 不超过设定”）
                        msk = det_map > tau_q
                        # 连通域过滤（去散点/去巨大块）
                        min_cc = int(vcfg.get("min_cc", vcfg.get("mask_min_size", 1)))
                        max_cc = vcfg.get("max_cc", vcfg.get("mask_max_size", None))
                        if isinstance(max_cc, (int, float)) and float(max_cc) > 0:
                            max_cc = int(max_cc)
                        else:
                            max_cc = None
                        msk = _filter_cc_by_size(msk, min_size=min_cc, max_size=max_cc)

                        heat = det_map.copy()
                        heat[~msk] = 0.0

                        # 可选：把 (tau,1] 线性拉伸到 (0,1]，让目标区域有更好的动态范围
                        if bool(vcfg.get("rescale", True)):
                            denom = max(1e-12, float(1.0 - tau_q))
                            heat = np.clip((heat - tau_q) / denom, 0.0, 1.0)

                        # 可选：中值滤波去散点（仅影响可视化）
                        med = int(vcfg.get("median", vcfg.get("median_k", 0)))
                        if med and med > 1:
                            heat = _median_filter01(heat, ksize=med)

                        # 可选：再做一次 pow 压缩（背景更暗）
                        g = float(vcfg.get("gamma", vcfg.get("pow_gamma", 1.0)))
                        if g and g != 1.0:
                            heat = np.power(np.clip(heat, 0, 1), g)
                    elif mode in ("binary", "pf_binary"):
                        pf_v = float(vcfg.get("pf", 1e-3))
                        tau_q = _tau_from_neg_pf(det_map, gt, pf_v)
                        msk = det_map > tau_q
                        min_cc = int(vcfg.get("min_cc", vcfg.get("mask_min_size", 1)))
                        max_cc = vcfg.get("max_cc", vcfg.get("mask_max_size", None))
                        if isinstance(max_cc, (int, float)) and float(max_cc) > 0:
                            max_cc = int(max_cc)
                        else:
                            max_cc = None
                        msk = _filter_cc_by_size(msk, min_size=min_cc, max_size=max_cc)
                        heat = msk.astype(np.float32)
                    else:
                        # unknown mode -> fallback to default det_vis
                        heat = det_vis

                    heat = np.asarray(heat, dtype=np.float32)
                    out_name = os.path.join(
                        result_dir, f"{Name}{suffix}_detMap_color_{tag}.png"
                    )
                    plt.figure()
                    plt.imshow(
                        heat, cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
                    )
                    plt.axis("off")
                    plt.savefig(out_name, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    print(
                        f"[viz{suffix}] saved viridis variant -> {os.path.basename(out_name)}"
                    )
                except Exception as ex:
                    print(
                        f"[viz{suffix}] WARN: failed to save variant {tag} ({mode}): {ex}"
                    )

        # --- 保存 ROC CSV ---
        if len(thr) > 1:
            np.savetxt(
                os.path.join(result_dir, f"{Name}{suffix}roc_3d.csv"),
                np.column_stack((thr, fpr, tpr)),
                delimiter=",",
                header="tau,pf,pd",
                comments="",
            )
            np.savetxt(
                os.path.join(result_dir, f"{Name}{suffix}roc_pf_pd.csv"),
                np.column_stack((fpr, tpr)),
                delimiter=",",
                header="pf,pd",
                comments="",
            )
            np.savetxt(
                os.path.join(result_dir, f"{Name}{suffix}roc_tau_pf.csv"),
                np.column_stack((thr, fpr)),
                delimiter=",",
                header="tau,pf",
                comments="",
            )
            np.savetxt(
                os.path.join(result_dir, f"{Name}{suffix}roc_tau_pd.csv"),
                np.column_stack((thr, tpr)),
                delimiter=",",
                header="tau,pd",
                comments="",
            )
            print(f"[eval{suffix}] ROC CSV 已保存到 result/")

        # --- PF 约束下的二值 mask + overlay（推荐 Sandiego 用） ---
        save_masks = bool(pcfg.get("save_pf_masks", True))
        if save_masks and len(thr) > 1:
            gt_bin = gt == 1
            overlay_gt = bool(pcfg.get("overlay_gt", True))
            min_cc = int(pcfg.get("mask_min_size", 1))

            for pf in pf_list:
                pf_f = float(pf)
                tau = _tau_for_pf(fpr, thr, pf_f)
                mask = det_map >= tau
                mask = _filter_small_cc(mask, min_size=min_cc)

                pf_tag = _fmt_pf(pf_f).replace("-", "m")
                mask_fname = os.path.join(
                    result_dir, f"{Name}{suffix}_mask_pf{pf_tag}.png"
                )
                plt.figure()
                plt.imshow(
                    mask.astype(np.uint8),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                plt.axis("off")
                plt.savefig(mask_fname, dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close()

                if bool(pcfg.get("save_overlay", True)):
                    ov_fname = os.path.join(
                        result_dir, f"{Name}{suffix}_overlay_pf{pf_tag}.png"
                    )
                    plt.figure()
                    plt.imshow(
                        bg_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
                    )
                    plt.imshow(
                        mask.astype(np.uint8),
                        cmap="Reds",
                        alpha=0.55,
                        interpolation="nearest",
                    )
                    if overlay_gt:
                        plt.imshow(
                            gt_bin.astype(np.uint8),
                            cmap="Greens",
                            alpha=0.35,
                            interpolation="nearest",
                        )
                    plt.axis("off")
                    plt.savefig(ov_fname, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

                    # 单独保存 GT 叠加图（方便论文排版）
                    ov2_fname = os.path.join(
                        result_dir, f"{Name}{suffix}_overlay_pf{pf_tag}_detOnly.png"
                    )
                    plt.figure()
                    plt.imshow(
                        bg_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
                    )
                    plt.imshow(
                        mask.astype(np.uint8),
                        cmap="Reds",
                        alpha=0.60,
                        interpolation="nearest",
                    )
                    plt.axis("off")
                    plt.savefig(ov2_fname, dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()

                print(
                    f"[viz{suffix}] PF={pf_f:g} -> tau={tau:.6f}  mask saved (min_cc={min_cc})"
                )

        # profile 完成后清 cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.empty_cache()
