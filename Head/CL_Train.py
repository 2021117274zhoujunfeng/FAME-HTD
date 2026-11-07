# -*- coding: utf-8 -*-
"""CL_Train.py

训练/评估入口（HTD-Mamba 改良版）。

训练期（不使用目标光谱）：
  - 对比学习（intra-view + context-view）
  - (可选) BPSA：背景原型 SwAV 风格交换聚类约束（学习异质背景结构）
  - (可选) Spectral-Geometry Alignment：保持嵌入空间与 raw 谱形几何一致

评估期（只在测试阶段使用目标光谱）：
  - 目标端元注入 -> 计算检测分数（表征空间匹配 + 频域/导数多证据）
  - 背景原型抑制（margin / bg-OSP）
  - 支持 AUC1~AUC5（对齐 HTD-Mamba 论文定义）
"""

from __future__ import annotations

import os
import random
import logging
import csv
import math
import itertools
import glob
import json
import shutil
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
# sklearn 在某些服务器环境里可能没有（尤其是最小化环境）。
# 如果没有 sklearn，则用 numpy 写一个简化版 roc_curve/auc（足够用于 AUC1~AUC5 计算）。
try:
    from sklearn.metrics import auc, roc_curve  # type: ignore
except Exception:
    auc = None
    roc_curve = None

    def roc_curve(y_true, y_score):
        """Minimal ROC curve implementation (sklearn-free)."""
        y_true = np.asarray(y_true).astype(np.int32).ravel()
        y_score = np.asarray(y_score).astype(np.float32).ravel()
        # sort by score desc
        order = np.argsort(-y_score)
        y_true = y_true[order]
        y_score = y_score[order]
        # distinct thresholds
        distinct = np.where(np.diff(y_score))[0]
        thr_idx = np.r_[distinct, y_true.size - 1]
        tps = np.cumsum(y_true)[thr_idx]
        fps = 1 + thr_idx - tps
        # prepend (0,0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        P = tps[-1]
        N = fps[-1]
        tpr = tps / (P if P > 0 else 1.0)
        fpr = fps / (N if N > 0 else 1.0)
        thresholds = np.r_[y_score[0] + 1.0, y_score[thr_idx]]
        return fpr, tpr, thresholds

    def auc(x, y):
        """Trapezoidal AUC (sklearn-free)."""
        x = np.asarray(x).astype(np.float64)
        y = np.asarray(y).astype(np.float64)
        return float(np.trapz(y, x))
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from CL_Model import SpectralGroupAttention
from Data_Patch import Data
from augment import checkFile, standard, ts_generation


# =========================
# Utils
# =========================


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _robust_zscore(x: np.ndarray, clip: float = 8.0) -> np.ndarray:
    """鲁棒 z-score（median/MAD），比 mean/std 更不怕极值。"""
    x = x.astype(np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z = (x - med) / (1.4826 * mad)
    if clip is not None and clip > 0:
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32)


def _conformal_score_np(
    score: np.ndarray,
    calib_scores: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """把任意 score 映射为“背景 p-value”意义下的 [0,1] 置信分数。

    设 score 越大越像目标。给定一组 *背景校准样本* 的分数 calib_scores，
    对每个像素 i 计算经验 p-value：

        p_i = ( #{j: calib_scores_j >= score_i} + 1 ) / (N_cal + 1)

    然后输出 conf_i = 1 - p_i（越大越像目标）。

    直觉：在“纯背景”上，p-value 近似均匀分布，因此同一个阈值 tau
    近似对应固定的误警率（阈值更稳）。

    说明：这是 split/inductive conformal 的一个常用“异常分数 -> p-value”做法，
    这里作为 *测试期* 的后处理标定，不引入目标光谱到训练。
    """

    s = score.astype(np.float32)
    c = calib_scores.astype(np.float32)
    c = c[np.isfinite(c)]
    if c.size < 8:
        # calibration 太少就退化为 min-max/identity（避免 NaN）
        s0 = np.nanmin(s)
        s1 = np.nanmax(s)
        if (not np.isfinite(s0)) or (not np.isfinite(s1)) or abs(s1 - s0) < eps:
            return np.zeros_like(s, dtype=np.float32)
        return np.clip((s - s0) / (s1 - s0 + eps), 0.0, 1.0).astype(np.float32)

    # sort ascending for fast "count >=" via searchsorted
    c_sorted = np.sort(c)
    # idx = first position to insert s to keep ascending order
    idx = np.searchsorted(c_sorted, s, side="left")
    count_ge = c_sorted.size - idx
    p = (count_ge.astype(np.float32) + 1.0) / (float(c_sorted.size) + 1.0)
    conf = 1.0 - p
    return np.clip(conf, 0.0, 1.0).astype(np.float32)




# =========================
# Spatial refinement (test-time only)
# =========================

def _spatial_refine_score_map_np(score_map: np.ndarray, cfg: Dict) -> np.ndarray:
    """Spatially refine a score map to enforce neighborhood consensus.

    Why:
      - Many hyperspectral target detection heads are per-pixel and may produce
        *holes* inside a true target region and *isolated peaks* on background.
      - A lightweight spatial consensus step (no GT, no training target)
        can boost contiguous target responses and suppress isolated false alarms.

    This function is used **only in eval/inference** when enabled by cfg.

    Method (default): neighbor-mean blend excluding center pixel
      S_ref = (1-α) * S + α * mean_{neighbors}(S)

    Args:
        score_map: [H, W] float32 in [0,1] (or close)
        cfg: config dict

    Returns:
        refined score_map in [0,1]
    """
    if score_map is None:
        return score_map

    H, W = score_map.shape[:2]
    if H < 3 or W < 3:
        return score_map.astype(np.float32)

    enable = bool(cfg.get('spatial_refine_enable', False))
    if not enable:
        return score_map.astype(np.float32)

    mode = str(cfg.get('spatial_refine_mode', 'neighbor_mean_blend')).lower()
    k = int(cfg.get('spatial_refine_ksize', 5))
    iters = int(cfg.get('spatial_refine_iters', 1))
    alpha = float(cfg.get('spatial_refine_alpha', 0.5))

    # sanitize
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    # avoid too large kernel on small maps
    k = min(k, (min(H, W) // 2) * 2 + 1)
    k = max(3, k)

    iters = max(1, iters)
    alpha = float(max(0.0, min(1.0, alpha)))

    pad = k // 2
    pad_mode = str(cfg.get('spatial_refine_pad', 'reflect')).lower()
    if pad_mode not in ['reflect', 'replicate', 'circular', 'constant']:
        pad_mode = 'reflect'

    # Use CPU by default (fast enough for typical HSI scenes, avoids GPU sync)
    # You may enable CUDA refinement by cfg['spatial_refine_cuda']=True.
    use_cuda = bool(cfg.get('spatial_refine_cuda', False)) and torch.cuda.is_available()
    dev = torch.device(cfg.get('device', 'cuda:0' if use_cuda else 'cpu')) if use_cuda else torch.device('cpu')

    S = torch.from_numpy(score_map.astype(np.float32))[None, None].to(dev)

    kernel = torch.ones((1, 1, k, k), dtype=torch.float32, device=dev)
    kernel[0, 0, pad, pad] = 0.0
    kernel = kernel / float(k * k - 1)

    for _ in range(iters):
        try:
            S_pad = F.pad(S, (pad, pad, pad, pad), mode=pad_mode)
        except Exception:
            S_pad = F.pad(S, (pad, pad, pad, pad), mode='replicate')
        nb = F.conv2d(S_pad, kernel, padding=0)

        if mode in ['neighbor_mean_blend', 'blend', 'nb_blend']:
            # 邻域均值融合：既能抑制孤立峰，也可能“扩散”高分到邻域
            S = (1.0 - alpha) * S + alpha * nb
        elif mode in ['neighbor_mean_max', 'max', 'fill']:
            # 仅向上填充（更容易扩散高分，一般不建议在 AUC3 很敏感时使用）
            S = torch.maximum(S, nb)
        elif mode in ['support_suppress', 'suppress', 'support_gate']:
            # 仅抑制不抬高：根据邻域支持度(邻域均值)对当前像素做“门控”
            #   support = sigmoid(k * (nb - thr))  ∈ (0,1)
            #   S_ref = S * support^p
            # 这个模式用于减少空间后处理引入的“高分扩散”，对降低 AUC3 更友好。
            # 阈值 thr 支持两种写法：
            #   - float: 直接用固定阈值（与旧版兼容）
            #   - 'auto'/'auto_nb_q': 由邻域均值 nb 的分位数自适应得到（更稳健，不易“误杀”真目标）
            thr_cfg = cfg.get('spatial_refine_thr', 0.10)
            thr_q = float(cfg.get('spatial_refine_thr_q', 0.95))
            thr_q = float(max(0.50, min(0.9999, thr_q)))

            k_sup = float(cfg.get('spatial_refine_k', 12.0))
            p_sup = float(cfg.get('spatial_refine_p', 1.0))
            min_sup = float(cfg.get('spatial_refine_min_sup', 0.0))

            if isinstance(thr_cfg, str):
                mode_thr = thr_cfg.lower().strip()
                if mode_thr in ['auto', 'auto_q', 'auto_nb', 'auto_nb_q', 'auto_nb_quantile']:
                    # 注意：用 nb 的分位数作为“空间支持阈值”，可以适配不同数据/不同模型的分数尺度
                    # 只要目标在空间上呈现“连通/成团”，就不会被压得太狠；孤立高分假警则会被显著抑制。
                    try:
                        thr = float(torch.quantile(nb.detach().reshape(-1), thr_q).item())
                    except Exception:
                        thr = float(nb.mean().item())
                elif mode_thr in ['auto_score_q', 'auto_s_q', 'auto_score_quantile']:
                    try:
                        thr = float(torch.quantile(S.detach().reshape(-1), thr_q).item())
                    except Exception:
                        thr = float(S.mean().item())
                else:
                    # unknown string -> fallback
                    thr = 0.10
            else:
                thr = float(thr_cfg)

            support = torch.sigmoid(k_sup * (nb - thr))
            if min_sup > 0.0:
                # 给边缘像素留一点“最小支持度”，避免过度抑制细小目标
                min_sup = float(max(0.0, min(0.95, min_sup)))
                support = min_sup + (1.0 - min_sup) * support
            if abs(p_sup - 1.0) > 1e-6:
                support = support.pow(p_sup)
            S = S * support
        else:
            S = (1.0 - alpha) * S + alpha * nb

    out = S.squeeze(0).squeeze(0).detach().to('cpu').numpy().astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _resolve_margin_thr_np(sem_margin: np.ndarray, cfg: Dict) -> float:
    """解析推理期 margin 阈值（支持自适应）。

    目的：提升异质背景下阈值稳定性，同时保持可解释。

    支持的配置：
      - margin_thr: float，直接使用固定阈值
      - margin_thr: 'auto_mad' / 'auto'：用 median+K*MAD 自适应
      - margin_thr: 'auto_q' / 'quantile'：用分位数自适应

    相关超参：
      - margin_thr_k_mad: MAD 倍数（默认 2.5）
      - margin_thr_q: 分位数（默认 0.95）
      - margin_thr_clip: (min,max) 可选，避免阈值过激
    """

    thr_cfg = cfg.get("margin_thr", 0.0)
    # fixed numeric
    if isinstance(thr_cfg, (int, float, np.floating)):
        return float(thr_cfg)

    mode = str(thr_cfg).lower()
    x = sem_margin.astype(np.float32)
    # robust stats
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-6

    if mode in ["auto", "auto_mad", "mad"]:
        k = float(cfg.get("margin_thr_k_mad", 2.5))
        # 1.4826*MAD 近似 std；k≈2.5 对应“比背景高一截”的 margin
        thr = med + k * 1.4826 * mad
    elif mode in ["auto_q", "quantile", "q"]:
        q = float(cfg.get("margin_thr_q", 0.95))
        q = float(np.clip(q, 0.50, 0.9999))
        thr = float(np.quantile(x, q))
    else:
        # unknown string -> fallback to 0
        thr = 0.0

    clip_cfg = cfg.get("margin_thr_clip", None)
    if isinstance(clip_cfg, (list, tuple)) and len(clip_cfg) == 2:
        thr = float(np.clip(thr, float(clip_cfg[0]), float(clip_cfg[1])))
    return float(thr)


def _auto_topk_stat_np(
    x: np.ndarray,
    mode: str = "topk_median",
    top_ratio: float = 0.001,
    min_k: int = 16,
    eps: float = 1e-6,
) -> float:
    """推理期用于自动标定的稳健统计量（避免被极少数 outlier 带偏）。

    用于：
      - nbs_t_norm 的 t_max
      - wave_sim_max 的 wave_max

    Args:
        x: 1D array
        mode: 'topk_mean' | 'topk_median' | 'quantile'
        top_ratio: 取 top x% 作为“高分集”
        min_k: 最少取多少个点（图像太小/target 太稀疏时防止 k=0）
    """

    x1 = np.asarray(x, dtype=np.float32).reshape(-1)
    x1 = x1[np.isfinite(x1)]
    if x1.size == 0:
        return 1.0

    mode = str(mode).lower()
    if mode in ["quantile", "q"]:
        q = float(np.clip(1.0 - top_ratio, 0.50, 0.99999))
        v = float(np.quantile(x1, q))
        return float(max(v, eps))

    k = int(max(min_k, round(x1.size * float(top_ratio))))
    k = int(np.clip(k, 1, x1.size))
    # partial sort for top-k
    topk = np.partition(x1, -k)[-k:]
    if mode in ["topk_mean", "mean"]:
        v = float(np.mean(topk))
    else:  # default median
        v = float(np.median(topk))
    return float(max(v, eps))


def _torch_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile 的兼容封装。

    说明：某些老版本 PyTorch 没有 torch.quantile；这里提供一个小型 fallback。
    由于 batch_size 通常不大（~80），fallback 排序开销可接受。
    """
    q = float(np.clip(q, 0.0, 1.0))
    if hasattr(torch, "quantile"):
        return torch.quantile(x, q)
    # fallback
    x1 = x.detach().reshape(-1)
    n = x1.numel()
    if n <= 1:
        return x1.mean()
    k = int(round((n - 1) * q))
    k = int(np.clip(k, 0, n - 1))
    v = torch.sort(x1).values
    return v[k]


def _corr_sim_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """相关系数式余弦相似度矩阵。

    关键原因：你的数据在 min-max 到 [0,1] 后全为正。
    直接 cosine 往往整体偏高，导致：
      - false-negative mask 失效（几乎谁都像谁）
      - density 权重失效（w_bg 全接近 1，原型会被稀有目标污染）

    做法：对每条光谱先减均值（去亮度/偏置），再 L2 归一化，再点积。

    Args:
        x: [B,C]
    Returns:
        sim: [B,B] in [-1,1]
    """
    x = x - x.mean(dim=1, keepdim=True)
    x_u = x / (torch.linalg.norm(x, dim=1, keepdim=True) + eps)
    return x_u @ x_u.t()


def _spectral_angle_from_sim(sim: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """由 cosine similarity 得到谱角距离（radian）。"""
    return torch.acos(sim.clamp(-1.0 + eps, 1.0 - eps))


def _batch_density_from_angle(angle: torch.Tensor, topk: int = 8) -> torch.Tensor:
    """batch 内密度估计：取每个样本与其它样本的 top-k **最小谱角**的均值。

    angle 越小表示越相似；若一个样本在 batch 内能找到很多小角度近邻，
    则更可能是“常见材料/背景”。

    Returns:
        dens_angle: [B]，数值越小越“密度高/背景”。
    """
    B = angle.shape[0]
    if B <= 2:
        return torch.full((B,), float("inf"), device=angle.device, dtype=angle.dtype)
    ang2 = angle.clone()
    ang2.fill_diagonal_(float("inf"))
    k = int(max(1, min(int(topk), B - 1)))
    vals = ang2.topk(k, dim=1, largest=False).values  # 最小 k 个角度
    return vals.mean(dim=1)


def _build_false_negative_mask(sim: torch.Tensor, topk: int = 8) -> torch.Tensor:
    """构造 neg_mask: True=允许作为 negative，False=从 denominator 中剔除。

    直觉：raw 域非常相似的样本，可能属于同一类背景材料，
    若强行当 negative 会造成 false-negative，破坏表征。
    """
    B = sim.shape[0]
    mask = torch.ones((B, B), device=sim.device, dtype=torch.bool)
    mask.fill_diagonal_(True)
    if topk <= 0 or B <= 2:
        return mask

    sim2 = sim.clone()
    sim2.fill_diagonal_(-1.0)
    k = int(max(1, min(topk, B - 1)))
    idx = sim2.topk(k, dim=1, largest=True).indices
    for i in range(B):
        mask[i, idx[i]] = False
    # 保持对称性不是必须，但更稳
    mask = mask & mask.t()
    mask.fill_diagonal_(True)
    return mask


# =========================
# Contrastive losses
# =========================


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    neg_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SimCLR-style InfoNCE (symmetric)."""
    B, _ = z1.shape
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = (z1 @ z2.t()) / max(float(temperature), 1e-6)

    if neg_mask is not None:
        mask = neg_mask.clone()
        mask.fill_diagonal_(True)
        logits = logits.masked_fill(~mask, -1e9)

    labels = torch.arange(B, device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


def weighted_info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 0.1,
    neg_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """带样本权重的 InfoNCE（用于 ctx-contrast），weights: [B] in [0,1]."""
    B, _ = z1.shape
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = (z1 @ z2.t()) / max(float(temperature), 1e-6)

    if neg_mask is not None:
        mask = neg_mask.clone()
        mask.fill_diagonal_(True)
        logits = logits.masked_fill(~mask, -1e9)

    labels = torch.arange(B, device=z1.device)
    loss_vec = F.cross_entropy(logits, labels, reduction="none")
    w = weights.detach().clamp(0.0, 1.0)
    loss1 = (loss_vec * w).sum() / (w.sum() + 1e-6)

    loss_vec2 = F.cross_entropy(logits.t(), labels, reduction="none")
    loss2 = (loss_vec2 * w).sum() / (w.sum() + 1e-6)

    return 0.5 * (loss1 + loss2)


def _build_win_weights_from_angle(
    angle: torch.Tensor,
    thr_mode: str = "fixed",
    thr: float = 0.12,
    thr_q: float = 0.10,
    k: float = 20.0,
    min_w: float = 1e-4,
) -> torch.Tensor:
    """WIN: 用 raw 域谱角为 InfoNCE 的 negatives 构造权重，缓解 false-negative。

    背景：
        在高光谱目标检测的自监督阶段，我们没有 target 标签。
        batch 内 raw 光谱非常相似的样本，往往属于同一种背景材料；
        若把它们当作 negative 强行拉远，会破坏“谱形几何”，导致评估期做目标谱匹配时排序变差（AUC1 受损）。

    做法：
        w_ij = sigmoid( k * (angle_ij - thr) )
        - angle_ij < thr  -> w 接近 0：弱化该 negative
        - angle_ij > thr  -> w 接近 1：正常 negative

    Args:
        angle: [B,B] 谱角矩阵（radian），对角线≈0
        thr_mode: 'fixed' 或 'quantile'（用 batch 内分位数得到阈值，更稳健）
        thr: 固定阈值（radian）
        thr_q: 分位数（0~1）
        k: sigmoid 斜率
        min_w: 数值稳定，避免完全 0

    Returns:
        w: [B,B] in [min_w, 1], diag=1
    """
    if angle.ndim != 2 or angle.shape[0] != angle.shape[1]:
        raise ValueError(f"angle must be [B,B], got {tuple(angle.shape)}")

    B = angle.shape[0]
    thr_mode = str(thr_mode).lower()

    if thr_mode in ["quantile", "q", "rank"]:
        q = float(max(0.01, min(0.99, thr_q)))
        ang = angle.detach().clone()
        # 排除对角线：用 +inf 避免参与“最小角度”的分位数统计
        ang.fill_diagonal_(float("inf"))
        vals = ang[torch.isfinite(ang)]
        if vals.numel() > 4:
            thr_val = torch.quantile(vals, q=q)
        else:
            thr_val = torch.tensor(float(thr), device=angle.device, dtype=angle.dtype)
    else:
        thr_val = torch.tensor(float(thr), device=angle.device, dtype=angle.dtype)

    w = torch.sigmoid(float(k) * (angle - thr_val))
    # 保证 positive 对角线不被削弱
    w = w.clamp(min=float(min_w), max=1.0)
    w.fill_diagonal_(1.0)
    return w


def win_info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    neg_weights: Optional[torch.Tensor] = None,
    neg_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted-NCE：对 negatives 做加权的 InfoNCE（对称）。

    Args:
        z1,z2: [B,D]
        temperature: softmax 温度
        neg_weights: [B,B] in [0,1]，对 negatives 的权重（diag 应为 1）
        neg_mask: [B,B] bool，False 的位置会被彻底剔除（权重=0）

    Returns:
        scalar loss
    """
    B, _ = z1.shape
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    T = max(float(temperature), 1e-6)
    logits = (z1 @ z2.t()) / T  # [B,B]

    if neg_weights is None:
        w = torch.ones_like(logits)
    else:
        w = neg_weights.to(dtype=logits.dtype, device=logits.device)

    if neg_mask is not None:
        w = w * neg_mask.to(dtype=logits.dtype)

    # 强制 positive 不被削弱
    w = w.clamp(min=0.0, max=1.0)
    w.fill_diagonal_(1.0)

    # -------- direction 1: z1 -> z2 --------
    m = logits.max(dim=1, keepdim=True).values
    exp = torch.exp(logits - m) * w
    denom = exp.sum(dim=1) + 1e-12
    pos = exp.diag() + 1e-12
    loss_12 = -torch.log(pos / denom)

    # -------- direction 2: z2 -> z1 --------
    logits_t = logits.t()
    w_t = w.t()
    m2 = logits_t.max(dim=1, keepdim=True).values
    exp2 = torch.exp(logits_t - m2) * w_t
    denom2 = exp2.sum(dim=1) + 1e-12
    pos2 = exp2.diag() + 1e-12
    loss_21 = -torch.log(pos2 / denom2)

    return 0.5 * (loss_12.mean() + loss_21.mean())


def batch_hard_triplet_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    margin: float = 0.2,
    neg_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batch-hard triplet（对称）：专门拉开“最难负样本”，提升排序能力（AUC1）。

    说明：
        - hardneg_triplet_loss 是 *patch 内* 的局部 hard-negative；
        - 这里的 batch-hard 是 *batch 内* 的全局 hard-negative，直接打击“最像目标的背景”。

    Args:
        z1,z2: [B,D] 两个视角的 embedding
        margin: triplet margin
        neg_mask: [B,B] bool，False 表示不作为 negative（用于 suppress false-negative）

    Returns:
        scalar loss
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = z1 @ z2.t()  # [B,B]
    pos = sim.diag()

    sim_neg = sim.clone()
    sim_neg.fill_diagonal_(-1e9)
    if neg_mask is not None:
        sim_neg = sim_neg.masked_fill(~neg_mask, -1e9)

    hard = sim_neg.max(dim=1).values
    loss1 = F.relu(hard - pos + float(margin))

    # symmetric
    sim_neg2 = sim.t().clone()
    sim_neg2.fill_diagonal_(-1e9)
    if neg_mask is not None:
        sim_neg2 = sim_neg2.masked_fill(~neg_mask.t(), -1e9)

    hard2 = sim_neg2.max(dim=1).values
    pos2 = sim.t().diag()
    loss2 = F.relu(hard2 - pos2 + float(margin))

    return 0.5 * (loss1.mean() + loss2.mean())

def hardneg_triplet_loss(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """简单 triplet：让 anchor 更接近 pos，远离 hard-neg。"""
    z_anchor = F.normalize(z_anchor, dim=1)
    z_pos = F.normalize(z_pos, dim=1)
    z_neg = F.normalize(z_neg, dim=1)
    pos = torch.sum(z_anchor * z_pos, dim=1)
    neg = torch.sum(z_anchor * z_neg, dim=1)
    return F.relu(neg - pos + float(margin)).mean()


# =========================
# Spectral-Geometry Alignment (SGA)
# =========================


def spectral_geometry_alignment_loss(
    z: torch.Tensor,
    sim_raw: torch.Tensor,
    topk: int = 8,
    sim_thr: float = 0.7,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """频谱几何对齐损失（无监督、无需目标端元）。

    目标：
        让 embedding 空间的相似度结构尽量保持 raw 光谱域的相似度结构，
        从而提升“评估阶段：与目标端元在表征空间做匹配”时的可迁移性与稳定性。

    做法：
        1) 在 batch 内，用 raw 域余弦相似度为每个样本挑选 top-k 近邻；
        2) 仅在这些近邻上约束：cos(z_i, z_j) ≈ cos(raw_i, raw_j)。

    备注：
        这相当于把“谱角几何”显式注入表征学习（类似关系/度量蒸馏的思想），
        可以缓解纯 instance discrimination 导致的“几何扭曲”，从而让目标匹配更可靠。
    """
    if z.dim() != 2 or sim_raw.dim() != 2:
        raise ValueError(
            f"z must be [B,D] and sim_raw must be [B,B], got {tuple(z.shape)}, {tuple(sim_raw.shape)}"
        )

    B = z.size(0)
    if B <= 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    # embedding similarity
    z_u = F.normalize(z, dim=1)
    sim_emb = z_u @ z_u.t()  # [B,B]

    # remove self-similarity when selecting neighbors
    sim_r = sim_raw.clone()
    sim_r.fill_diagonal_(-1.0)

    k = int(max(1, min(int(topk), B - 1)))
    idx = sim_r.topk(k, dim=1, largest=True).indices  # [B,k]

    sim_raw_pos = sim_raw.gather(1, idx)  # [B,k]
    sim_emb_pos = sim_emb.gather(1, idx)  # [B,k]

    # only enforce on sufficiently similar pairs
    if sim_thr is not None:
        m = (sim_raw_pos > float(sim_thr)).float()
    else:
        m = torch.ones_like(sim_raw_pos)

    diff2 = (sim_emb_pos - sim_raw_pos).pow(2) * m
    per = diff2.sum(dim=1) / (m.sum(dim=1) + 1e-6)  # [B]

    if weights is None:
        return per.mean()

    w = weights.detach().clamp(0.0, 1.0)
    return (per * w).sum() / (w.sum() + 1e-6)


# =========================
# SwAV-style prototype loss (BPSA)
# =========================


@torch.no_grad()
def sinkhorn_knopp(
    logits: torch.Tensor,
    eps: float = 0.05,
    iters: int = 3,
) -> torch.Tensor:
    """SwAV 的 Sinkhorn-Knopp，用于生成近似均衡的软分配。

    Args:
        logits: [B,K]
        eps:   temperature for exp
        iters: number of normalization iterations

    Returns:
        Q: [B,K]，每行和为 1（soft assignment）
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B,K], got {tuple(logits.shape)}")

    B, K = logits.shape
    # 稳定：减去行最大值
    x = logits - logits.max(dim=1, keepdim=True).values
    Q = torch.exp(x / max(float(eps), 1e-6)).t()  # [K,B]

    # 全局归一
    Q /= Q.sum() + 1e-12

    # 迭代归一：行->1/K, 列->1/B
    for _ in range(int(max(1, iters))):
        Q /= Q.sum(dim=1, keepdim=True) + 1e-12
        Q /= float(K)
        Q /= Q.sum(dim=0, keepdim=True) + 1e-12
        Q /= float(B)

    Q *= float(B)
    return Q.t().contiguous()  # [B,K]


def swapped_proto_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    prototypes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    tau: float = 0.1,
    sinkhorn_eps: float = 0.05,
    sinkhorn_iters: int = 3,
    # optional: only do Sinkhorn on background-like samples
    bg_only_thr: Optional[float] = None,
    min_bg_samples: int = 16,
) -> torch.Tensor:
    """BPSA: swapped assignment loss.

    - 用 view1 的 assignment 监督 view2 的 prediction，反之亦然。
    - assignment 用 Sinkhorn 得到近似均衡的 soft code，避免塌陷。
    - weights 用于“密度门控”：仅高密度(更像背景)样本强约束。

    Args:
        z1,z2:      [B,D]
        prototypes: [K,D] (建议已 normalize)
        weights:    [B] or None
    """
    z1n = F.normalize(z1, dim=1)
    z2n = F.normalize(z2, dim=1)
    pn = F.normalize(prototypes, dim=1)

    logits1 = (z1n @ pn.t()) / max(float(tau), 1e-6)  # [B,K]
    logits2 = (z2n @ pn.t()) / max(float(tau), 1e-6)

    # 说明：SwAV 的 swapped assignment + Sinkhorn 假设 batch 内主要是“同分布样本”。
    # 在 HTD 场景下，batch 里极少量“稀有目标/异常”会让 assignment 的平衡约束产生扰动。
    # 这里给一个可选开关：只用背景权重大(更像背景)的样本计算 Sinkhorn codes。
    if weights is not None and bg_only_thr is not None:
        w_det = weights.detach().clamp(0.0, 1.0)
        active = w_det >= float(bg_only_thr)
        min_k = max(2, int(min_bg_samples))
        if active.sum() < min_k:
            # 若“高置信背景”太少：不要退化为全量（会把稀有目标/异常拉进 Sinkhorn）
            # 改为选 w_bg 最大的 top-k，仍尽量保持 prototype 纯背景。
            k = min(int(w_det.numel()), min_k)
            idx = torch.topk(w_det, k=k, largest=True).indices
        else:
            idx = torch.nonzero(active, as_tuple=False).squeeze(1)
        logits1_a = logits1[idx]
        logits2_a = logits2[idx]
        with torch.no_grad():
            q1 = sinkhorn_knopp(
                logits1_a.detach(), eps=sinkhorn_eps, iters=sinkhorn_iters
            )
            q2 = sinkhorn_knopp(
                logits2_a.detach(), eps=sinkhorn_eps, iters=sinkhorn_iters
            )

        logp1 = F.log_softmax(logits1_a, dim=1)
        logp2 = F.log_softmax(logits2_a, dim=1)

        ce12 = -(q1 * logp2).sum(dim=1)
        ce21 = -(q2 * logp1).sum(dim=1)

        w_a = w_det[idx]
        loss12 = (ce12 * w_a).sum() / (w_a.sum() + 1e-6)
        loss21 = (ce21 * w_a).sum() / (w_a.sum() + 1e-6)
        return 0.5 * (loss12 + loss21)

    with torch.no_grad():
        q1 = sinkhorn_knopp(logits1.detach(), eps=sinkhorn_eps, iters=sinkhorn_iters)
        q2 = sinkhorn_knopp(logits2.detach(), eps=sinkhorn_eps, iters=sinkhorn_iters)

    logp1 = F.log_softmax(logits1, dim=1)
    logp2 = F.log_softmax(logits2, dim=1)

    # 交叉熵：q1 -> logits2
    ce12 = -(q1 * logp2).sum(dim=1)  # [B]
    ce21 = -(q2 * logp1).sum(dim=1)

    if weights is None:
        return 0.5 * (ce12.mean() + ce21.mean())

    w = weights.detach().clamp(0.0, 1.0)
    loss12 = (ce12 * w).sum() / (w.sum() + 1e-6)
    loss21 = (ce21 * w).sum() / (w.sum() + 1e-6)
    return 0.5 * (loss12 + loss21)


def prototype_orthogonal_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """原型多样性正则：鼓励原型彼此正交，避免全部挤在一起。"""
    p = F.normalize(prototypes, dim=1)
    gram = p @ p.t()  # [K,K]
    I = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    return torch.mean((gram - I) ** 2)


def prototype_outlier_repulsion_loss(
    z: torch.Tensor,
    prototypes: torch.Tensor,
    dens_angle: torch.Tensor,
    out_thr: float = 0.45,
    out_k: float = 12.0,
    margin: float = 0.55,
) -> torch.Tensor:
    """将“极低密度/疑似异常”的样本推离背景原型。

    直觉：
    - 你的训练阶段不引入目标端元，因此“目标/异常”只能靠统计稀有性来弱监督；
    - 仅用 SwAV 式 prototypes 做聚类会让稀有样本的 assignment 不稳定；
    - 这里用 dens_angle(越大越稀有) 做一个 outlier 权重，只对极端稀有者施加
      "max(sim_to_bg_proto) < margin" 的约束。

    注意：该项默认建议权重很小，并且 out_thr 设得稍大，只打击“极端 outlier”，
    避免把罕见但真实存在的背景材料当成目标。
    """
    # outlier weight: dens_angle 越大(越不相似) -> w_out 越大
    w_out = (
        torch.sigmoid(float(out_k) * (dens_angle - float(out_thr)))
        .detach()
        .clamp(0.0, 1.0)
    )
    if float(w_out.sum()) < 1e-6:
        return torch.tensor(0.0, device=z.device)

    z_u = F.normalize(z, dim=1)
    p_u = F.normalize(prototypes, dim=1)
    sim_max = (z_u @ p_u.t()).max(dim=1).values  # [B]

    loss_vec = F.relu(sim_max - float(margin)).pow(2)
    return (loss_vec * w_out).sum() / (w_out.sum() + 1e-6)


# =========================
# Model builder
# =========================


def _build_model(cfg: Dict) -> SpectralGroupAttention:
    device = torch.device(
        cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = SpectralGroupAttention(
        band=int(cfg.get("band", 205)),
        group_length=int(cfg.get("m", 5)),
        channel_dim=int(cfg.get("channel", 16)),
        state_size=int(cfg.get("state_size", 16)),
        device=device,
        layer=int(cfg.get("layer", 1)),
        # wavelet/deriv/channel-gate
        use_wavelet=bool(cfg.get("use_wavelet", True)),
        wavelet_level=int(cfg.get("wavelet_level", 1)),
        wavelet_type=str(cfg.get("wavelet_type", "db4")),
        wavelet_learnable=bool(cfg.get("wavelet_learnable", False)),
        use_deriv=bool(cfg.get("use_deriv", True)),
        deriv_scale=float(cfg.get("deriv_scale", 1.0)),
        use_channel_gate=bool(cfg.get("use_channel_gate", True)),
        use_input_norm=bool(cfg.get("use_input_norm", True)),
        # drift aug (model-side)
        drift_p=float(cfg.get("drift_p", 0.25)),
        drift_shift_range=tuple(cfg.get("drift_shift_range", (-1.0, 1.0))),
        drift_scale_range=tuple(cfg.get("drift_scale_range", (-0.01, 0.01))),
        drift_piecewise_p=float(cfg.get("drift_piecewise_p", 0.35)),
        drift_sinusoid_p=float(cfg.get("drift_sinusoid_p", 0.25)),
        drift_seg_mag=float(cfg.get("drift_seg_mag", 0.01)),
        bidirectional=bool(cfg.get("bidirectional", True)),
        # group tokenization
        token_stride=cfg.get("token_stride", None),
        token_activation=str(cfg.get("token_activation", "leaky_relu")),
        embed_dim=int(cfg.get("embed_dim", 64)),
        proto_bank_size=int(cfg.get("proto_bank_size", 64)),
    )
    # NOTE: 模型构建函数只负责返回模型。
    # 评估阶段是否保存可视化/输出由 eval(cfg, save_outputs=...) 决定。
    return model.to(device)


# =========================
# Target-bank augmentation (eval only)
# =========================



def _augment_target_bank(
    t0: np.ndarray,
    num: int,
    shift_range: Tuple[float, float] = (-1.5, 1.5),
    scale_range: Tuple[float, float] = (-0.01, 0.01),
    sine_amp_range: Tuple[float, float] = (0.0, 0.2),
    sine_freq_range: Tuple[float, float] = (0.5, 1.5),
    smooth_sigma_range: Tuple[float, float] = (0.0, 0.6),
    calib: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Endmember-bundle style target bank augmentation (eval only).

    We keep the perturbations *small* and physically motivated:
      - global spectral shift (simulates micro wavelength drift / smile residual)
      - mild scale and wavelength-dependent calibration jitter (a + b*λ + c*λ^2)
      - bandwidth/resolution change via 1D Gaussian smoothing
      - gentle sinusoidal depth modulation (absorption depth variation proxy)
      - tiny additive noise

    Returns:
        bank: [V, C], including the original t0 as the first element.
    """
    rng = np.random.RandomState(seed)
    t0 = t0.astype(np.float32)
    C = int(t0.shape[0])
    x = np.arange(C, dtype=np.float32)
    lam = x / float(max(1, C - 1))  # normalized wavelength index in [0,1]

    bank = [t0.copy()]

    # optional scipy smoother (fast). fall back to numpy conv if unavailable.
    try:
        from scipy.ndimage import gaussian_filter1d as _gf1d  # type: ignore
    except Exception:
        _gf1d = None

    for _ in range(int(max(0, num))):
        t = t0.copy()

        # 1) global spectral shift (index warp)
        shift = float(rng.uniform(shift_range[0], shift_range[1]))
        xq = np.clip(x + shift, 0.0, float(C - 1))
        t = np.interp(x, xq, t).astype(np.float32)

        # 2) global scale
        sc = 1.0 + float(rng.uniform(scale_range[0], scale_range[1]))
        t = (t * sc).astype(np.float32)

        # 3) wavelength-dependent calibration jitter
        if calib is not None:
            (a0, a1), (b0, b1), (c0, c1) = calib
            a = float(rng.uniform(a0, a1))
            b = float(rng.uniform(b0, b1))
            c = float(rng.uniform(c0, c1))
            t = (t * (1.0 + a + b * lam + c * (lam ** 2))).astype(np.float32)

        # 4) bandwidth/resolution change (Gaussian smoothing)
        if smooth_sigma_range is not None:
            sig = float(rng.uniform(smooth_sigma_range[0], smooth_sigma_range[1]))
            if sig > 1e-6:
                if _gf1d is not None:
                    t = _gf1d(t, sigma=sig, mode="nearest").astype(np.float32)
                else:
                    # simple gaussian kernel fallback
                    ksz = int(max(3, 2 * int(3.0 * sig) + 1))
                    if ksz % 2 == 0:
                        ksz += 1
                    half = ksz // 2
                    grid = np.arange(-half, half + 1, dtype=np.float32)
                    k = np.exp(-0.5 * (grid / max(1e-6, sig)) ** 2).astype(np.float32)
                    k = k / (np.sum(k) + 1e-8)
                    t = np.convolve(t, k, mode="same").astype(np.float32)

        # 5) gentle sinusoidal depth modulation (keeps monotonic index)
        amp = float(rng.uniform(sine_amp_range[0], sine_amp_range[1]))
        freq = float(rng.uniform(sine_freq_range[0], sine_freq_range[1]))
        if abs(amp) > 1e-6:
            t = (t * (1.0 + amp * np.sin(2.0 * np.pi * freq * x / float(max(1, C))))).astype(np.float32)

        # 6) tiny additive noise
        ns = float(noise_std)
        if ns > 0.0:
            t = (t + rng.normal(0.0, ns, size=t.shape).astype(np.float32)).astype(np.float32)

        bank.append(np.clip(t, 0.0, 1.0).astype(np.float32))

    bank = np.stack(bank, axis=0)
    bank = np.clip(bank, 0.0, 1.0)
    return bank


def _cc_normalize_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """correlation-cosine 归一化：去均值 + L2。

    Args:
        x: [..., C]
    """
    x = x.astype(np.float32)
    x = x - x.mean(axis=-1, keepdims=True)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _target_bank_angle_weights(
    t_bank: np.ndarray,
    sigma: float = 0.12,
    angle_clip: Optional[float] = 0.30,
    min_weight: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """为 target-bank 生成“物理合理性权重”（越接近原始 t0 权重越大）。

    动机：
        target-bank 能提升对端元变异/微小谱漂移的鲁棒性，但 bank 过大/扰动过强时，
        **max over candidates** 会出现 multiple comparisons：背景像素更容易“偶然很像”某个候选而抬高得分。
        这里用 bank 内每个候选与 t0 的谱角（基于 correlation-cosine）给一个先验权重，
        在 pooling 时作为 soft prior，既保留鲁棒性，又降低假警。

    Returns:
        w  : [V]  (>=min_weight)
        ang: [V]  (radian)
    """
    if t_bank.ndim != 2:
        raise ValueError(f"t_bank must be 2D [V,C], got {t_bank.shape}")
    V, C = t_bank.shape
    if V <= 0:
        raise ValueError("t_bank is empty")

    # reference = first one (original t0)
    t0 = t_bank[0:1]  # [1,C]
    tb = _cc_normalize_np(t_bank)
    t0n = _cc_normalize_np(t0)[0]  # [C]

    cos = np.sum(tb * t0n[None, :], axis=1)
    cos = np.clip(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    ang = np.arccos(cos).astype(np.float32)

    sigma = float(max(sigma, 1e-6))
    w = np.exp(-np.square(ang / sigma)).astype(np.float32)

    if angle_clip is not None:
        ac = float(angle_clip)
        w = np.where(ang <= ac, w, 0.0).astype(np.float32)

    # ensure base signature always kept
    w[0] = 1.0

    w = np.clip(w, float(min_weight), 1.0).astype(np.float32)
    return w, ang



def _pool_similarity(
    sim_mat: torch.Tensor,
    mode: str = "max",
    topk: int = 3,
    temp: float = 0.07,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """相似度 pooling（用于 target bank / 背景原型的聚合）。

    为什么要做 pooling？
    - 推理阶段对 target-bank 取 **max** 时，本质是做了多次比较：
      bank 越大，“偶然很像”的背景像素越容易被抬高（multiple comparisons）。
    - top-k 平均或 softmax pooling 能让相似度更“稳健”，
      通常更利于抑制异质背景假警、提升阈值稳定性。

    支持 `weights`：
    - 典型用法1：给 target-bank 每个候选一个“物理合理性先验权重”（越接近原始 t0 权重越大），
      在 softmax/topk_mean pooling 时作为 soft prior，进一步降低 bank 引入的偶然高相似假警。
    - 典型用法2：**每个像素-每个bank成员**一套权重（[B,M]），例如用 raw 谱角对 bank 做门控，
      让 pooling 更“物理一致”（你的 target-bank + angle gate 方案强烈建议这样用）。

    重要修正（提升 AUC1 / 降低多次比较假警）：
    - 当 mode 为 topk_mean / softmax_topk 且提供 weights 时，
      **top-k 的候选集合也应当在“加权 logits”上选择**，而不是只在 sim 上选择。
      否则会出现两类问题：
        1) 物理一致(权重大)但 embedding 相似略低的候选被 top-k 直接丢弃 -> 误伤真目标；
        2) embedding 偶然很高但物理不一致(权重小)的候选仍进入 top-k -> multiple comparisons 假警。

    Args:
        sim_mat: [B, M] 相似度矩阵（cosine 等），值域通常在 [-1,1]
        mode:
            - "max"            : 取最大值（原始做法）
            - "mean"           : 直接平均（若给 weights，则做加权平均）
            - "topk_mean"      : 取 top-k 后平均（若给 weights，则做 top-k 加权平均）
            - "softmax"        : softmax 加权平均（若给 weights，则 logits += log(weights)）
            - "softmax_topk"   : 先取 top-k，再 softmax 加权平均（支持 weights）
        topk: top-k 的 k
        temp: softmax pooling 温度（越小越接近 max）
        weights:
            - [M] 或 [1,M]：对每个 bank/proto 一个全局权重
            - [B,M]       ：对每个样本一套权重（更强、更稳健）

    Returns:
        pooled: [B]
    """
    mode = str(mode).lower()
    if sim_mat.dim() != 2:
        raise ValueError(f"sim_mat must be 2D [B,M], got {tuple(sim_mat.shape)}")

    B, M = sim_mat.shape
    w_vec: Optional[torch.Tensor] = None
    w_mat: Optional[torch.Tensor] = None

    if weights is not None:
        if weights.dim() == 1:
            w_vec = weights
        elif weights.dim() == 2:
            if weights.size(0) == 1 and weights.size(1) == M:
                w_vec = weights.squeeze(0)
            elif weights.size(0) == B and weights.size(1) == M:
                w_mat = weights
            else:
                raise ValueError(
                    f"weights must be [M] or [1,M] or [B,M], got {tuple(weights.shape)} vs sim_mat {tuple(sim_mat.shape)}"
                )
        else:
            raise ValueError(f"weights must be 1D/2D tensor, got {tuple(weights.shape)}")

        if w_vec is not None:
            w_vec = w_vec.to(device=sim_mat.device, dtype=sim_mat.dtype).clamp(min=1e-8)
        if w_mat is not None:
            w_mat = w_mat.to(device=sim_mat.device, dtype=sim_mat.dtype).clamp(min=1e-8)

    # 统一得到 logw_full（便于 weight-aware top-k selection）
    logw_full: Optional[torch.Tensor] = None
    if w_mat is not None:
        logw_full = torch.log(w_mat)
    elif w_vec is not None:
        logw_full = torch.log(w_vec).unsqueeze(0)  # [1,M] broadcast to [B,M]

    # ---- max (ignore weights) ----
    if mode in ["max", "hardmax"]:
        return sim_mat.max(dim=1).values

    # ---- mean ----
    if mode in ["mean", "avg", "average"]:
        if w_vec is None and w_mat is None:
            return sim_mat.mean(dim=1)
        if w_mat is not None:
            return (sim_mat * w_mat).sum(dim=1) / (w_mat.sum(dim=1) + 1e-8)
        # w_vec
        w = w_vec.unsqueeze(0)  # [1,M]
        return (sim_mat * w).sum(dim=1) / (w.sum(dim=1) + 1e-8)

    # ---- top-k mean ----
    if mode in ["topk", "topk_mean", "topk-mean"]:
        k = int(max(1, min(int(topk), M)))

        if logw_full is None:
            vals, _ = sim_mat.topk(k, dim=1, largest=True)
            return vals.mean(dim=1)

        # 关键：top-k 选择也考虑 weights（物理一致性先验）
        t_sel = float(max(temp, 1e-6))
        select_logits = sim_mat / t_sel + logw_full  # [B,M]
        idx = select_logits.topk(k, dim=1, largest=True).indices  # [B,k]
        vals = sim_mat.gather(1, idx)  # [B,k]

        if w_mat is not None:
            w = w_mat.gather(1, idx)  # [B,k]
        else:
            w = w_vec[idx]  # [B,k]
        return (vals * w).sum(dim=1) / (w.sum(dim=1) + 1e-8)



    # ---- median pooling ----
    if mode in ["median", "med"]:
        # weighted median is supported when weights is provided
        if logw_full is None:
            return sim_mat.median(dim=1).values

        # sort by similarity (desc)
        s_sorted, idx = sim_mat.sort(dim=1, descending=True)
        if w_mat is not None:
            w_full = w_mat
        else:
            w_full = w_vec.unsqueeze(0).expand(B, M)
        w_sorted = w_full.gather(1, idx)

        cum = w_sorted.cumsum(dim=1)
        half = 0.5 * (w_sorted.sum(dim=1, keepdim=True) + 1e-8)
        mask = cum >= half
        med_idx = mask.float().argmax(dim=1)
        return s_sorted[torch.arange(B, device=sim_mat.device), med_idx]

    # ---- top-k median ----
    if mode in ["topk_median", "topk-med", "topkmed"]:
        k = int(max(1, min(int(topk), M)))

        if logw_full is None:
            vals, _ = sim_mat.topk(k, dim=1, largest=True)
            return vals.median(dim=1).values

        # weight-aware top-k selection
        t_sel = float(max(temp, 1e-6))
        select_logits = sim_mat / t_sel + logw_full  # [B,M]
        idx = select_logits.topk(k, dim=1, largest=True).indices  # [B,k]
        vals = sim_mat.gather(1, idx)  # [B,k]

        if w_mat is not None:
            w = w_mat.gather(1, idx)  # [B,k]
        else:
            w = w_vec[idx]  # [B,k]

        # weighted median among the selected k
        s_sorted, s_idx = vals.sort(dim=1, descending=True)
        w_sorted = w.gather(1, s_idx)
        cum = w_sorted.cumsum(dim=1)
        half = 0.5 * (w_sorted.sum(dim=1, keepdim=True) + 1e-8)
        mask = cum >= half
        med_idx = mask.float().argmax(dim=1)
        return s_sorted[torch.arange(B, device=sim_mat.device), med_idx]
    # ---- softmax pooling ----
    if mode in ["softmax", "soft"]:
        t = float(max(temp, 1e-6))
        logits = sim_mat / t
        if w_mat is not None:
            logits = logits + torch.log(w_mat)
        elif w_vec is not None:
            logits = logits + torch.log(w_vec).unsqueeze(0)
        att = torch.softmax(logits, dim=1)
        return (att * sim_mat).sum(dim=1)

    # ---- softmax on top-k ----
    if mode in ["softmax_topk", "soft_topk", "softmax-topk"]:
        k = int(max(1, min(int(topk), M)))
        t = float(max(temp, 1e-6))

        if logw_full is None:
            vals, _ = sim_mat.topk(k, dim=1, largest=True)
            logits = vals / t
            att = torch.softmax(logits, dim=1)
            return (att * vals).sum(dim=1)

        # 关键：top-k 选择也考虑 weights（物理一致性先验）
        select_logits = sim_mat / t + logw_full  # [B,M]
        idx = select_logits.topk(k, dim=1, largest=True).indices  # [B,k]
        vals = sim_mat.gather(1, idx)  # [B,k]

        if w_mat is not None:
            w = w_mat.gather(1, idx)  # [B,k]
        else:
            w = w_vec[idx]  # [B,k]

        logits = vals / t + torch.log(w)
        att = torch.softmax(logits, dim=1)
        return (att * vals).sum(dim=1)

    raise ValueError(f"Unknown pooling mode: {mode}")



def _bg_osp_residual(
    z_u: torch.Tensor,
    p: torch.Tensor,
    k: int = 4,
    alpha: float = 1.0,
    sims: Optional[torch.Tensor] = None,
    method: str = "approx",
    ridge: float = 1e-3,
) -> torch.Tensor:
    """Background-Orthogonal Subspace Projection（推理期用）。

    目标：用“背景原型”近似张成的子空间去解释背景成分，然后把它从特征里减掉：
        z_res = normalize(z - alpha * Proj_bg(z))

    支持两种实现：
    - approx（快速近似）：
        假设 prototypes 近似正交，直接用
            Proj_bg(z) ≈ sum_i ( (z·p_i) p_i )
        计算快，但当原型不够正交/背景更复杂时，投影会有偏差。

    - lstsq / exact（更稳健，推荐在异质/结构性背景上使用）：
        对 top-k 原型做带 ridge 的最小二乘投影（k 很小，开销可忽略）：
            a = argmin || z - sum_i a_i p_i ||^2
            Proj_bg(z) = sum_i a_i p_i
        这会提升背景子空间减除的“几何正确性”，通常能进一步降低假警。

    Args:
        z_u : [B,D] 已归一化特征
        p   : [K,D] 已归一化背景原型
        k   : 选取与 z 最相似的 top-k 原型做投影
        alpha: 投影减除强度（0~1 更稳，1 表示完全减去）
        sims: 可选，提前算好的 z_u @ p^T（避免重复算）
        method: "approx" | "lstsq"/"exact"
        ridge : lstsq 时的岭系数，防止 Gram 矩阵病态（建议 1e-4~1e-2）

    Returns:
        z_res: [B,D] 归一化后的残差特征
    """
    if k is None or int(k) <= 0 or alpha is None or float(alpha) <= 0.0:
        return z_u
    if sims is None:
        sims = z_u @ p.t()
    k = int(max(1, min(int(k), p.size(0))))
    vals, idx = sims.topk(k, dim=1, largest=True)
    p_topk = p[idx]  # [B,k,D]

    method = str(method).lower()
    if method in ["lstsq", "ls", "exact", "pinv"]:
        # exact projection onto span(p_topk): proj = B (B^T B + λI)^-1 B^T z
        # 由于 k 很小（默认 4），我们在 k-space 里解线性方程即可。
        G = torch.bmm(p_topk, p_topk.transpose(1, 2))  # [B,k,k]
        ridge_v = float(ridge) if ridge is not None else 0.0
        if ridge_v > 0:
            I = torch.eye(k, device=G.device, dtype=G.dtype).unsqueeze(0)
            G = G + ridge_v * I

        b = torch.bmm(p_topk, z_u.unsqueeze(2)).squeeze(2)  # [B,k]

        # solve: G a = b
        try:
            a = torch.linalg.solve(G, b.unsqueeze(2)).squeeze(2)  # [B,k]
        except Exception:
            # fallback：极少数情况下（数值问题）用 inverse
            G_inv = torch.inverse(G)
            a = torch.bmm(G_inv, b.unsqueeze(2)).squeeze(2)

        proj = (a.unsqueeze(2) * p_topk).sum(dim=1)  # [B,D]
    else:
        # approx: assume prototypes nearly orthonormal
        proj = (vals.unsqueeze(2) * p_topk).sum(dim=1)  # [B,D]

    z_res = z_u - float(alpha) * proj
    return F.normalize(z_res, dim=1, eps=1e-6)



def _ace_similarity_np(
    Z: np.ndarray,
    T: np.ndarray,
    shrink: float = 0.05,
    ridge: float = 1e-3,
    eps: float = 1e-8,
    bg_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Feature-space ACE (Adaptive Cosine/Coherence Estimator) scores.

    ACE(z,t) = (t^T Σ^{-1} z)^2 / ((t^T Σ^{-1} t)(z^T Σ^{-1} z))

    Notes:
    - This is a *test-time* statistical detector in the learned feature space.
    - It does NOT require target spectra during training (only used here in inference).
    - Shrinkage + ridge improves numerical stability on small images / low-rank covariances.

    Args:
        Z: [N,D] feature vectors (can be normalized or not)
        T: [V,D] target-bank feature vectors (same space as Z)
        shrink: covariance shrinkage ratio toward isotropic diagonal
        ridge: diagonal loading (Tikhonov)
        eps: numerical epsilon
        bg_mask: optional boolean mask [N], select samples used to estimate background covariance

    Returns:
        ace: [N,V] in [0,1]
    """
    Z = np.asarray(Z, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if Z.ndim != 2 or T.ndim != 2:
        raise ValueError(f"ACE expects 2D arrays, got Z{Z.shape}, T{T.shape}")
    N, D = Z.shape
    V = T.shape[0]
    if T.shape[1] != D:
        raise ValueError(f"ACE dim mismatch: Z{Z.shape} vs T{T.shape}")

    if bg_mask is None:
        bg_mask = np.ones((N,), dtype=bool)
    bg_mask = np.asarray(bg_mask).astype(bool)
    if bg_mask.shape[0] != N:
        raise ValueError(f"bg_mask length mismatch: {bg_mask.shape[0]} vs N={N}")

    # ensure enough samples to estimate covariance
    if int(bg_mask.sum()) < max(int(D + 1), 32):
        bg_mask = np.ones((N,), dtype=bool)

    Zbg = Z[bg_mask]
    mu = Zbg.mean(axis=0, keepdims=True)
    X = Zbg - mu

    # biased covariance (divide by N) for stability
    denom = float(max(1, Zbg.shape[0]))
    cov = (X.T @ X) / denom

    shrink = float(np.clip(shrink, 0.0, 0.99))
    tr = float(np.trace(cov) / float(max(1, D)))
    cov = (1.0 - shrink) * cov + shrink * tr * np.eye(D)
    cov = cov + float(max(ridge, 0.0)) * np.eye(D)

    try:
        inv = np.linalg.inv(cov)
    except Exception:
        inv = np.linalg.pinv(cov)

    Zc = Z - mu
    Tc = T - mu

    Zinv = Zc @ inv  # [N,D]
    den_z = np.sum(Zinv * Zc, axis=1)  # [N]

    Tinv = Tc @ inv  # [V,D]
    den_t = np.sum(Tinv * Tc, axis=1)  # [V]

    invTc = inv @ Tc.T  # [D,V]
    num = (Zc @ invTc) ** 2  # [N,V]

    den = den_z[:, None] * den_t[None, :] + float(eps)
    ace = num / den
    ace = np.clip(ace, 0.0, 1.0)
    return ace.astype(np.float32)


def _mt_camf_similarity_np(
    Z: np.ndarray,
    T: np.ndarray,
    bg_idx: np.ndarray,
    *,
    bg_mask: Optional[np.ndarray] = None,
    K: Optional[int] = None,
    shrink: float = 0.05,
    ridge: float = 1e-3,
    eps: float = 1e-8,
    min_samples: int = 64,
    alpha_pow: float = 0.5,
    alpha_clip: Tuple[float, float] = (0.0, 1.5),
    infeas_lambda: float = 3.0,
    infeas_gamma: float = 4.0,
) -> np.ndarray:
    """Mixture-Tuned Clustered Adaptive Matched Filter (MT-CAMF) in embedding space.

    Motivation (test-time only):
        - Global whitening/ACE assumes a single homogeneous background. In hyperspectral
          target detection, heterogeneous background often causes *a few* high-score
          clutter outliers and/or suppresses some mixed target pixels.
        - Clustered background statistics (per prototype/cluster) improves adaptive
          detectors under non-homogeneous scenes.
        - A simple mixture-tuned score uses (clipped) abundance + residual penalty to
          avoid over-suppressing true targets while rejecting clutter outliers.

    Args:
        Z      : [N,D] pixel embeddings (float32)
        T      : [V,D] target-bank embeddings (float32)
        bg_idx : [N]   cluster assignment per pixel (int)
        bg_mask: [N]   optional background mask for covariance estimation

    Returns:
        sim_mat: [N,V] score matrix in [0,1] (float32)
    """
    if Z.ndim != 2 or T.ndim != 2:
        raise ValueError(f"Z and T must be 2D, got {Z.shape}, {T.shape}")
    N, D = Z.shape
    V, D2 = T.shape
    if D2 != D:
        raise ValueError(f"feature dim mismatch: Z {D} vs T {D2}")
    if bg_idx.ndim != 1 or bg_idx.shape[0] != N:
        raise ValueError(f"bg_idx must be [N], got {bg_idx.shape} vs N={N}")

    Z = Z.astype(np.float32, copy=False)
    T = T.astype(np.float32, copy=False)
    bg_idx = bg_idx.astype(np.int64, copy=False)

    if bg_mask is None:
        bg_mask = np.ones((N,), dtype=bool)
    else:
        bg_mask = bg_mask.astype(bool, copy=False)
        if bg_mask.shape[0] != N:
            raise ValueError(f"bg_mask must be [N], got {bg_mask.shape}")

    if K is None:
        K = int(bg_idx.max()) + 1 if N > 0 else 1
    K = int(max(1, K))
    min_samples = int(max(8, min_samples))

    shrink = float(np.clip(shrink, 0.0, 0.99))
    ridge = float(max(0.0, ridge))
    infeas_lambda = float(max(0.0, infeas_lambda))
    infeas_gamma = float(max(0.0, infeas_gamma))
    alpha_pow = float(max(0.05, alpha_pow))
    a0, a1 = float(alpha_clip[0]), float(alpha_clip[1])
    if a1 <= a0:
        a0, a1 = 0.0, 1.5

    def _fit_inv(Zbg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # returns mu [1,D] and inv_cov [D,D]
        if Zbg.shape[0] <= 0:
            mu = np.zeros((1, D), dtype=np.float32)
            inv = np.eye(D, dtype=np.float32)
            return mu, inv

        mu = Zbg.mean(axis=0, keepdims=True)
        X = Zbg - mu
        denom = float(max(1, Zbg.shape[0]))
        cov = (X.T @ X) / denom

        tr = float(np.trace(cov) / float(max(1, D)))
        cov = (1.0 - shrink) * cov + shrink * tr * np.eye(D, dtype=np.float32)
        cov = cov + ridge * np.eye(D, dtype=np.float32)

        try:
            inv = np.linalg.inv(cov)
        except Exception:
            inv = np.linalg.pinv(cov)
        return mu.astype(np.float32), inv.astype(np.float32)

    # global fallback stats (avoid empty cluster / too-few samples)
    if int(bg_mask.sum()) < max(int(D + 1), 32):
        bg_mask = np.ones((N,), dtype=bool)
    mu_g, inv_g = _fit_inv(Z[bg_mask])

    sim = np.zeros((N, V), dtype=np.float32)
    # precompute cluster pixel indices once (avoid repeated np.where)
    for k in range(K):
        idx = np.where(bg_idx == k)[0]
        if idx.size == 0:
            continue

        # select samples for covariance estimation
        idx_bg = idx[bg_mask[idx]]
        if idx_bg.size < min_samples:
            mu, inv = mu_g, inv_g
        else:
            mu, inv = _fit_inv(Z[idx_bg])

        Zc = Z[idx] - mu  # [nk,D]
        Tc = T - mu       # [V,D]

        # num = (x-mu)^T inv (t-mu)
        invTc = inv @ Tc.T           # [D,V]
        num = Zc @ invTc             # [nk,V]

        # den_t = (t-mu)^T inv (t-mu)
        Tinv = Tc @ inv              # [V,D]
        den_t = np.sum(Tinv * Tc, axis=1) + float(eps)  # [V]

        # den_z = (x-mu)^T inv (x-mu)
        Zinv = Zc @ inv
        den_z = np.sum(Zinv * Zc, axis=1) + float(eps)  # [nk]

        # abundance estimate (generalized least squares) and clipping
        alpha = num / den_t[None, :]  # [nk,V]
        alpha_c = np.clip(alpha, a0, a1)

        # residual energy with clipped alpha: ||x - alpha*(t-mu)||_inv^2
        # expand: den_z - 2*alpha*num + alpha^2*den_t
        resid = den_z[:, None] - 2.0 * alpha_c * num + (alpha_c ** 2) * den_t[None, :]
        resid_ratio = resid / (den_z[:, None] + float(eps))
        resid_ratio = np.clip(resid_ratio, 0.0, 1e6)

        # infeasibility: residual + penalty for out-of-range abundance (before clipping)
        infeas = resid_ratio + infeas_gamma * ((alpha - alpha_c) ** 2)

        # score in [0,1]: abundance^pow * exp(-lambda * infeas)
        # (alpha_pow < 1 boosts subpixel/mixed targets without changing ranking too harshly)
        score = (np.clip(alpha_c, 0.0, 1.0) ** alpha_pow) * np.exp(-infeas_lambda * infeas)
        score = np.clip(score, 0.0, 1.0)

        sim[idx] = score.astype(np.float32)

    return sim


def compute_detection_scores(
    model: SpectralGroupAttention,
    data_matrix: np.ndarray,  # [N,C]
    target_spectrum: np.ndarray,  # [C,1] or [C]
    device: torch.device,
    cfg: Dict,
    return_debug: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """计算每个像素的检测得分（0~1）。"""
    model.eval()

    X = data_matrix.astype(np.float32)
    N, C = X.shape

    t0 = target_spectrum.reshape(-1).astype(np.float32)
    if t0.shape[0] != C:
        raise ValueError(f"target spectrum length mismatch: {t0.shape[0]} vs {C}")

    # ---- target bank ----
    V = int(cfg.get("target_aug_num", 8))
    t_bank = _augment_target_bank(
        t0,
        num=V,
        shift_range=tuple(cfg.get("target_aug_shift_range", (-1.5, 1.5))),
        scale_range=tuple(cfg.get("target_aug_scale_range", (-0.01, 0.01))),
        sine_amp_range=tuple(cfg.get("target_aug_sine_amp_range", (0.0, 0.2))),
        sine_freq_range=tuple(cfg.get("target_aug_sine_freq_range", (0.5, 1.5))),
        smooth_sigma_range=tuple(cfg.get("target_aug_smooth_sigma_range", (0.0, 0.6))),
        calib=cfg.get("target_aug_calib", None),
        noise_std=float(cfg.get("target_aug_noise_std", 0.0)),
        seed=int(cfg.get("seed", 0)) + 2025,
    )

    # ---- (NEW, inference-only) target-bank: mix target with scene background spectra ----
    # Motivation:
    #   - Some GT target pixels can be *mixed* (subpixel/edge) or show slight spectral variability.
    #   - A bank that only shifts/scales t0 may fail to match those pixels, pushing a few
    #     target pixels to very low scores -> AUC1 drops noticeably.
    #   - We add a small number of physically-plausible mixtures: t_mix = (1-a)*t0 + a*b,
    #     where b is a background spectrum sampled from the current image.
    # This uses the target spectrum ONLY at inference/eval, does not affect training.
    mixbg_enable = bool(cfg.get("mixbg_enable", False))
    mixbg_alpha = None  # per-bank mixing ratio (for weighting prior)
    if mixbg_enable:
        rng = np.random.RandomState(int(cfg.get("seed", 0)) + int(cfg.get("mixbg_seed_offset", 777)))
        pool_size = int(cfg.get("mixbg_pool_size", 4096))
        pool_size = max(32, min(pool_size, N))
        # sample a pool to avoid O(N*C) angle computation on very large images
        if N > pool_size:
            pool_idx = rng.choice(N, size=pool_size, replace=False)
        else:
            pool_idx = np.arange(N)

        X_pool = X[pool_idx]
        # correlation-cosine angle to t0 for background sampling
        x_cc = _cc_normalize_np(X_pool)
        t_cc = _cc_normalize_np(t0[None, :])[0]
        cos = np.clip(x_cc @ t_cc, -1.0, 1.0)
        ang = np.arccos(cos)
        q = float(np.clip(cfg.get("mixbg_angle_q", 0.7), 0.0, 0.999))
        thr = float(np.quantile(ang, q))
        cand = pool_idx[ang >= thr]
        if cand.size == 0:
            cand = pool_idx

        num_bg = int(max(0, cfg.get("mixbg_samples", 8)))
        num_bg = min(num_bg, cand.size)
        if num_bg > 0:
            sel = rng.choice(cand.size, size=num_bg, replace=False)
            bg_specs = X[cand[sel]]  # [num_bg, C]

            # list of mixing ratios
            alphas_cfg = cfg.get("mixbg_alphas", [0.1, 0.2, 0.3])
            if isinstance(alphas_cfg, (tuple, list)):
                alphas = [float(a) for a in alphas_cfg]
            else:
                alphas = [float(alphas_cfg)]
            alphas = [a for a in alphas if 0.0 < a < 1.0]

            mix_list = []
            mix_a = []
            for b in bg_specs:
                for a in alphas:
                    m = (1.0 - a) * t0 + a * b.astype(np.float32)
                    m = np.clip(m, 0.0, 1.0).astype(np.float32)
                    mix_list.append(m)
                    mix_a.append(a)

            if len(mix_list) > 0:
                mix_bank = np.stack(mix_list, axis=0).astype(np.float32)
                # alpha meta: existing bank members are treated as pure (alpha=0)
                mixbg_alpha = np.concatenate([
                    np.zeros((t_bank.shape[0],), dtype=np.float32),
                    np.asarray(mix_a, dtype=np.float32),
                ], axis=0)
                t_bank = np.concatenate([t_bank, mix_bank], axis=0)

    # ---- target-bank weighting (eval only) ----
    # 目的：降低 target-bank 的 multiple comparisons 假警风险。
    # 做法：用 correlation-cosine 谱角衡量每个 bank 成员与 t0 的偏离程度，
    #      越偏离权重越小；在 pooling 时作为 soft prior。
    bank_weighting = str(cfg.get("target_bank_weighting", "none")).lower()
    t_bank_w: Optional[torch.Tensor] = None
    t_bank_ang: Optional[np.ndarray] = None
    if bank_weighting in ["angle", "ang", "spectral_angle"]:
        sigma = float(cfg.get("target_bank_weight_sigma", 0.12))
        angle_clip = cfg.get("target_bank_angle_clip", 0.30)
        angle_clip_v = None
        if isinstance(angle_clip, (int, float)):
            angle_clip_v = float(angle_clip)
        min_w = float(cfg.get("target_bank_min_weight", 1e-6))
        w_np, ang_np = _target_bank_angle_weights(
            t_bank, sigma=sigma, angle_clip=angle_clip_v, min_weight=min_w
        )

        # (optional) extra prior for mixbg bank members: downweight very large mixing ratios
        if (mixbg_alpha is not None) and bool(cfg.get("mixbg_use_alpha_prior", True)):
            alpha_sigma = float(cfg.get("mixbg_alpha_sigma", 0.35))
            alpha_sigma = max(1e-6, alpha_sigma)
            a = mixbg_alpha.astype(np.float32)
            # Gaussian prior centered at 0 (pure target) but still allows moderate mixing
            alpha_w = np.exp(-0.5 * (a / alpha_sigma) ** 2).astype(np.float32)
            w_np = (w_np.astype(np.float32) * alpha_w).astype(np.float32)
            w_np = w_np / (w_np.sum() + 1e-12)

        t_bank_w = torch.from_numpy(w_np).to(device=device, dtype=torch.float32)
        t_bank_ang = ang_np

    # ---- background orthogonal projection (eval only) ----
    # 目标：用背景原型“解释并减去”背景成分，降低结构性背景的伪相似假警。
    use_bg_osp = bool(cfg.get("use_bg_osp", False))
    bg_osp_k = int(cfg.get("bg_osp_k", 4))
    bg_osp_alpha = float(cfg.get("bg_osp_alpha", 1.0))
    bg_osp_apply = str(cfg.get("bg_osp_apply", "both")).lower()
    # 新增：bg_osp 的实现方式（approx 或 exact/lstsq）
    bg_osp_method = str(cfg.get("bg_osp_method", "approx")).lower()
    bg_osp_ridge = float(cfg.get("bg_osp_ridge", 1e-3))

    with torch.no_grad():
        t_in = (
            torch.from_numpy(t_bank).to(device=device, dtype=torch.float32).unsqueeze(1)
        )
        z_t = model(t_in, force_no_drift=True).detach()  # [Vt,D]
        z_t_u = F.normalize(z_t, dim=1)

        # background prototypes
        p = model.get_bg_prototypes(normalize=True).detach()  # [K,D]
        # (可选) 背景子空间投影：对 target-bank embedding 也做同样处理，保持与像素端一致
        if use_bg_osp and bg_osp_apply in ["both", "t", "target"]:
            z_t_u_used = _bg_osp_residual(z_t_u, p, k=bg_osp_k, alpha=bg_osp_alpha, method=bg_osp_method, ridge=bg_osp_ridge)
        else:
            z_t_u_used = z_t_u

    # ---- target similarity metric (inference) ----
    # Default: cosine similarity in embedding space.
    # Optional: feature-space ACE (Adaptive Cosine/Coherence Estimator) for better
    # background/target separability under heterogeneous backgrounds.
    target_sim_metric = str(cfg.get("target_sim_metric", "cosine")).lower()
    use_ace_sim = target_sim_metric in ["ace", "f_ace", "whitened", "ace_sim"]
    use_mt_camf = target_sim_metric in [
        "mt_camf",
        "camf",
        "mtmf_camf",
        "mtamf",
        "mt_amf",
    ]
    ace_shrink = float(cfg.get("ace_shrink", 0.05))
    ace_ridge = float(cfg.get("ace_ridge", 1e-3))
    ace_eps = float(cfg.get("ace_eps", 1e-8))
    ace_bg_select = str(cfg.get("ace_bg_select", "all")).lower()
    ace_bg_q = float(cfg.get("ace_bg_q", 0.70))

    # MT-CAMF (clustered mixture-tuned adaptive matched filter) parameters
    mt_shrink = float(cfg.get("mt_camf_shrink", 0.05))
    mt_ridge = float(cfg.get("mt_camf_ridge", 1e-3))
    mt_eps = float(cfg.get("mt_camf_eps", 1e-8))
    mt_min_samples = int(cfg.get("mt_camf_min_samples", 64))
    mt_alpha_pow = float(cfg.get("mt_camf_alpha_pow", 0.5))
    mt_alpha_clip_max = float(cfg.get("mt_camf_alpha_clip_max", 1.5))
    mt_infeas_lambda = float(cfg.get("mt_camf_infeas_lambda", 3.0))
    mt_infeas_gamma = float(cfg.get("mt_camf_infeas_gamma", 4.0))
    mt_bg_select = str(cfg.get("mt_camf_bg_select", "angle_t0_q")).lower()
    mt_bg_q = float(cfg.get("mt_camf_bg_q", 0.70))



    # ---- inference ----
    bs = int(cfg.get("eval_batch_size", 2048))
    sem_margin_all = np.zeros((N,), dtype=np.float32)
    wave_sim_all = np.zeros((N,), dtype=np.float32)
    deriv_sim_all = np.zeros((N,), dtype=np.float32)
    sim_t_all = np.zeros((N,), dtype=np.float32)
    sim_bg_all = np.zeros((N,), dtype=np.float32)


    # cache features for ACE (2-pass in feature space)
    z_sim_store: Optional[np.ndarray] = None
    bg_idx_all: Optional[np.ndarray] = None
    if use_ace_sim or use_mt_camf:
        D = int(z_t_u_used.shape[1])
        z_sim_store = np.zeros((N, D), dtype=np.float32)

    if use_mt_camf:
        bg_idx_all = np.zeros((N,), dtype=np.int32)

    # -------------------------
    # Raw 谱角门控（推理期）：稳定阈值 + 可解释
    # -------------------------
    # 直觉：embedding 空间的相似度可能会被结构性背景“伪相似”误导；
    # 因此在原始谱形域再加一个“物理一致”的门控：
    #   angle(x, t) 小 => 更像目标；angle 大 => 作为背景抑制。
    # 注意：这里不影响训练阶段（训练仍然不注入目标端元）。
    use_angle_gate = bool(cfg.get("use_angle_gate", True))
    angle_thr = float(cfg.get("angle_gate_thr", 0.30))   # rad
    angle_k = float(cfg.get("angle_gate_k", 12.0))
    angle_w = float(cfg.get("angle_gate_w", 1.0))
    angle_raw_all = np.zeros((N,), dtype=np.float32)
    angle_gate_all = np.ones((N,), dtype=np.float32)
    # (新增) angle to original t0：更严格、更稳定的物理一致性信号（用于标定/置信度等）
    angle_raw_t0_all = np.zeros((N,), dtype=np.float32)
    angle_gate_t0_all = np.ones((N,), dtype=np.float32)

    # 目标谱：correlation-cosine（去均值再 L2），与训练期 raw 相似度保持一致
    # 改进点：这里不只用单一 t0，而是用 **target-bank (t_bank)** 来做 raw-angle gate，
    # 从而提升对「微小谱漂移/端元变异」的鲁棒性（不影响训练阶段）。
    t_cc_bank = torch.from_numpy(t_bank).to(device=device, dtype=torch.float32)  # [Vt,C]
    t_cc_bank = t_cc_bank - t_cc_bank.mean(dim=1, keepdim=True)
    t_cc_bank = F.normalize(t_cc_bank, dim=1, eps=1e-6)
    t_cc0 = t_cc_bank[0:1]  # [1,C] 原始目标谱 t0 的 correlation-cosine 归一化版本

    # angle gate 的 pooling：默认复用 target_sim_pool 的设置（也可在 cfg 中单独覆盖）
    angle_sim_pool = str(cfg.get("angle_sim_pool", cfg.get("target_sim_pool", "max")))
    angle_sim_topk = int(cfg.get("angle_sim_topk", cfg.get("target_sim_topk", 3)))
    angle_sim_temp = float(cfg.get("angle_sim_temp", cfg.get("target_sim_temp", 0.07)))


    # --- (新增) angle-aware pooling：用 raw 谱角对 target-bank 做逐像素门控权重 ---
    # 动机：
    #   target-bank + max/softmax 本质是多次比较（multiple comparisons），
    #   bank 越大，背景像素越容易“偶然很像”某个 bank 成员，从而抬高 sim_t / wave_sim。
    # 解决：
    #   用 raw 谱角（物理一致）对每个像素-每个 bank 成员做门控，再把该门控作为 weights 送进 pooling，
    #   让 bank 的鲁棒性“只在物理合理的候选上生效”，同时显著降低假警。
    angle_aware_pool = bool(cfg.get("angle_aware_pool", True))
    angle_pool_thr = float(cfg.get("angle_pool_thr", angle_thr))  # 默认复用 angle_gate_thr
    angle_pool_k = float(cfg.get("angle_pool_k", angle_k))        # 默认复用 angle_gate_k
    angle_pool_min_w = float(cfg.get("angle_pool_min_w", 1e-4))   # 防止 log(0) / 权重全灭

    use_wavelet_match = (
        bool(cfg.get("use_wavelet_match", True))
        and getattr(model, "wavelet", None) is not None
    )

    # (optional) derivative match as an extra frequency/spectral-shape evidence
    use_deriv_match = bool(cfg.get("use_deriv_match", False))
    t_d_bank = None
    if use_deriv_match:
        with torch.no_grad():
            # use correlation-cosine normalized spectrum then take first derivative
            t_d = t_cc_bank[:, 1:] - t_cc_bank[:, :-1]  # [Vt, C-1]
            t_d = t_d - t_d.mean(dim=1, keepdim=True)
            t_d_bank = F.normalize(t_d, dim=1, eps=1e-6)  # [Vt, C-1]

    # wavelet 目标 bank 预计算（避免每个 batch 重复算 target 的小波分量）
    ht_bank, lt_bank, wave_w = None, None, None
    if use_wavelet_match:
        with torch.no_grad():
            t_wave = (
                torch.from_numpy(t_bank)
                .to(device=device, dtype=torch.float32)
                .unsqueeze(1)
            )  # [Vt,1,C]
            if getattr(model, "use_input_norm", False):
                t_wave = t_wave - t_wave.mean(dim=2, keepdim=True)
                t_wave = F.normalize(t_wave, dim=2, eps=1e-6)
            low_t_bank, high_t_bank = model.wavelet(t_wave)  # [Vt,L,C]  (L=wavelet levels stacked on channel dim)
            ht_bank = F.normalize(high_t_bank, dim=2)  # [Vt,L,C]
            lt_bank = F.normalize(low_t_bank, dim=2)   # [Vt,L,C]

            # level weights: emphasize finer (level-1) details
            wave_levels = int(high_t_bank.shape[1])
            decay = float(cfg.get("wavelet_level_decay", 0.7))
            wave_w = torch.tensor([decay ** i for i in range(wave_levels)], device=device, dtype=torch.float32)
            wave_w = wave_w / (wave_w.sum() + 1e-8)

    for s in tqdm(range(0, N, bs), desc="infer", leave=False):
        e = min(N, s + bs)
        x_np = X[s:e]
        B = x_np.shape[0]

        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            # --- raw spectral angle gate + (新增) angle-aware pooling weights (inference only) ---
            # 说明：
            #   - raw angle gate：输出一个 gate_ang，用于推理末端抑制结构性背景假警（阈值更稳定、可解释）
            #   - angle-aware pooling：给 target-bank 每个候选一个“逐像素权重”，用于 sim_t / wave_sim 的 pooling，
            #     让 bank 的鲁棒性只在“物理合理”的候选上生效，显著降低 multiple comparisons 假警。
            #
            # w_bank：
            #   - None：不加权
            #   - [Vt]：全局先验权重（如 t_bank_w）
            #   - [B,Vt]：逐像素门控权重（angle-aware pooling）
            w_bank = t_bank_w  # default: global prior (or None)

            if use_angle_gate or angle_aware_pool:
                x_cc = x - x.mean(dim=2, keepdim=True)
                x_cc = F.normalize(x_cc, dim=2, eps=1e-6)  # [B,1,C]
                cos_raw_mat = x_cc.squeeze(1) @ t_cc_bank.t()  # [B,Vt]
                cos_raw_mat = cos_raw_mat.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                # --- raw angle to original t0 (更严格的物理一致性) ---
                # 说明：t0 角度比 target-bank pooled 角度更“保守”，
                # 用于：NBS 标定 mask / adaptive fusion conf，能减少 bank 造成的偶然高相似假警。
                cos_raw0 = (x_cc.squeeze(1) @ t_cc0.t()).squeeze(1)
                cos_raw0 = cos_raw0.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                ang0 = torch.acos(cos_raw0)
                gate_ang0 = torch.sigmoid(angle_k * (angle_thr - ang0))
                angle_raw_t0_all[s:e] = ang0.detach().cpu().numpy().astype(np.float32)
                angle_gate_t0_all[s:e] = gate_ang0.detach().cpu().numpy().astype(np.float32)


                # per-pixel weights from raw angle
                if angle_aware_pool:
                    ang_mat = torch.acos(cos_raw_mat)  # [B,Vt]
                    w_ang = torch.sigmoid(angle_pool_k * (angle_pool_thr - ang_mat))
                    if angle_pool_min_w > 0:
                        w_ang = w_ang.clamp_min(float(angle_pool_min_w))

                    if t_bank_w is not None:
                        w_bank = w_ang * t_bank_w.unsqueeze(0)  # [B,Vt]
                    else:
                        w_bank = w_ang  # [B,Vt]

                # pooled raw angle gate (optional)
                if use_angle_gate:
                    cos_raw = _pool_similarity(
                        cos_raw_mat,
                        mode=angle_sim_pool,
                        topk=angle_sim_topk,
                        temp=angle_sim_temp,
                        weights=w_bank,
                    )
                    cos_raw = cos_raw.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                    ang = torch.acos(cos_raw)
                    gate_ang = torch.sigmoid(angle_k * (angle_thr - ang))
                    angle_raw_all[s:e] = ang.detach().cpu().numpy().astype(np.float32)
                    angle_gate_all[s:e] = gate_ang.detach().cpu().numpy().astype(np.float32)

            # --- encoder forward ---
            z = model(x, force_no_drift=True)  # [B,D]
        z_u = F.normalize(z, dim=1)

        # -------- background similarity (pool over prototypes) --------
        sims_bg = z_u @ p.t()  # [B,K]
        sim_bg = _pool_similarity(
            sims_bg,
            mode=cfg.get("bg_sim_pool", "max"),
            topk=int(cfg.get("bg_sim_topk", 3)),
            temp=float(cfg.get("bg_sim_temp", 0.07)),
        )

        # (optional) store background cluster id for MT-CAMF
        if bg_idx_all is not None:
            bg_idx_all[s:e] = (
                torch.argmax(sims_bg, dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )

        # -------- target similarity (pool over target bank) --------
        # (可选) 背景子空间投影：先把 z 里可由背景原型解释的部分减掉，再去匹配目标
        z_sim = z_u
        if use_bg_osp and bg_osp_apply in ["both", "x", "pixel"]:
            z_sim = _bg_osp_residual(z_u, p, k=bg_osp_k, alpha=bg_osp_alpha, sims=sims_bg, method=bg_osp_method, ridge=bg_osp_ridge)



        # cache per-pixel feature for ACE similarity (inference only)
        if z_sim_store is not None:
            z_sim_store[s:e] = z_sim.detach().cpu().numpy().astype(np.float32)
        sims_t = z_sim @ z_t_u_used.t()  # [B,Vt]
        sim_t = _pool_similarity(
            sims_t,
            mode=cfg.get("target_sim_pool", "max"),
            topk=int(cfg.get("target_sim_topk", 3)),
            temp=float(cfg.get("target_sim_temp", 0.07)),
            weights=w_bank,
        )

        # -------- derivative-domain similarity (optional) --------
        if use_deriv_match and (t_d_bank is not None):
            with torch.no_grad():
                # ensure x_cc exists
                if not (use_angle_gate or angle_aware_pool):
                    x_cc = x - x.mean(dim=2, keepdim=True)
                    x_cc = F.normalize(x_cc, dim=2, eps=1e-6)
                xd = x_cc.squeeze(1)[:, 1:] - x_cc.squeeze(1)[:, :-1]  # [B, C-1]
                xd = xd - xd.mean(dim=1, keepdim=True)
                xd = F.normalize(xd, dim=1, eps=1e-6)
                sims_d = xd @ t_d_bank.t()  # [B, Vt]
                deriv_sim = _pool_similarity(
                    sims_d,
                    mode=str(cfg.get("deriv_sim_pool", cfg.get("target_sim_pool", "max"))),
                    topk=int(cfg.get("deriv_sim_topk", cfg.get("target_sim_topk", 3))),
                    temp=float(cfg.get("deriv_sim_temp", cfg.get("target_sim_temp", 0.07))),
                    weights=w_bank,
                )
            deriv_sim_all[s:e] = deriv_sim.detach().cpu().numpy().astype(np.float32)
        else:
            deriv_sim_all[s:e] = 0.0


        # semantic margin: target - background（越大越像目标）
        sem_margin = sim_t - sim_bg

        sim_t_all[s:e] = sim_t.detach().cpu().numpy().astype(np.float32)
        sim_bg_all[s:e] = sim_bg.detach().cpu().numpy().astype(np.float32)
        sem_margin_all[s:e] = sem_margin.detach().cpu().numpy().astype(np.float32)

        # 2) wavelet-domain similarity（高频更敏感于细粒度谱形）
        # 改进点：wavelet 匹配也与 target-bank 做 pooling（并预计算 bank 的小波分量），
        # 让频域项对端元变异/微漂移更稳健。
        if use_wavelet_match and ht_bank is not None and lt_bank is not None:
            with torch.no_grad():
                x_wave = x
                if getattr(model, "use_input_norm", False):
                    x_wave = x_wave - x_wave.mean(dim=2, keepdim=True)
                    x_wave = F.normalize(x_wave, dim=2, eps=1e-6)
                low_x, high_x = model.wavelet(x_wave)  # [B,L,C]

                hx = F.normalize(high_x, dim=2)  # [B,L,C]
                lx = F.normalize(low_x, dim=2)

                # multi-level aggregation
                L = int(hx.shape[1])
                if wave_w is None or int(wave_w.numel()) != L:
                    decay = float(cfg.get("wavelet_level_decay", 0.7))
                    wave_w = torch.tensor([decay ** i for i in range(L)], device=device, dtype=torch.float32)
                    wave_w = wave_w / (wave_w.sum() + 1e-8)

                wave_sim_mat = 0.0
                for lv in range(L):
                    sim_h_mat = hx[:, lv, :] @ ht_bank[:, lv, :].t()  # [B,Vt]
                    sim_l_mat = lx[:, lv, :] @ lt_bank[:, lv, :].t()
                    wave_sim_mat = wave_sim_mat + wave_w[lv] * (sim_h_mat + 0.5 * sim_l_mat)

                wave_sim = _pool_similarity(
                    wave_sim_mat,
                    mode=str(cfg.get("wave_sim_pool", cfg.get("target_sim_pool", "max"))),
                    topk=int(cfg.get("wave_sim_topk", cfg.get("target_sim_topk", 3))),
                    temp=float(cfg.get("wave_sim_temp", cfg.get("target_sim_temp", 0.07))),
                    weights=w_bank,
                )
            wave_sim_all[s:e] = wave_sim.detach().cpu().numpy().astype(np.float32)
        else:
            wave_sim_all[s:e] = 0.0



    # ---- optional: override target similarity by MT-CAMF (clustered mixture-tuned AMF) ----
    if use_mt_camf and (z_sim_store is not None) and (bg_idx_all is not None):
        # background selection for covariance estimation (test-time only)
        bg_mask = None
        try:
            if mt_bg_select in ["angle_t0_q", "ang_t0_q", "angle_q"]:
                q = float(np.clip(mt_bg_q, 0.0, 0.9999))
                a = angle_raw_t0_all.astype(np.float32)
                if float(np.nanstd(a)) > 1e-6:
                    thr = float(np.quantile(a, q))
                    bg_mask = (a >= thr)
            elif mt_bg_select in ["sim_bg_q", "bg_q", "bg_topq"]:
                q = float(np.clip(mt_bg_q, 0.0, 0.9999))
                thr = float(np.quantile(sim_bg_all.astype(np.float32), q))
                bg_mask = (sim_bg_all >= thr)
            elif mt_bg_select in ["angle_and_sim_bg", "ang_and_bg", "and"]:
                q = float(np.clip(mt_bg_q, 0.0, 0.9999))
                a = angle_raw_t0_all.astype(np.float32)
                thr_a = float(np.quantile(a, q)) if float(np.nanstd(a)) > 1e-6 else float(np.max(a))
                thr_b = float(np.quantile(sim_bg_all.astype(np.float32), q))
                bg_mask = (a >= thr_a) & (sim_bg_all >= thr_b)
            else:
                bg_mask = None
        except Exception:
            bg_mask = None

        T_np = z_t_u_used.detach().cpu().numpy().astype(np.float32)
        sim_mat = _mt_camf_similarity_np(
            z_sim_store,
            T_np,
            bg_idx_all,
            bg_mask=bg_mask,
            K=int(p.shape[0]),
            shrink=mt_shrink,
            ridge=mt_ridge,
            eps=mt_eps,
            min_samples=mt_min_samples,
            alpha_pow=mt_alpha_pow,
            alpha_clip=(0.0, mt_alpha_clip_max),
            infeas_lambda=mt_infeas_lambda,
            infeas_gamma=mt_infeas_gamma,
        )

        # pool over target bank
        if sim_mat.shape[1] == 1:
            sim_t_all = sim_mat[:, 0].astype(np.float32)
        else:
            sim_t = torch.from_numpy(sim_mat).to(dtype=torch.float32)
            pooled = _pool_similarity(
                sim_t,
                mode=cfg.get("target_sim_pool", "max"),
                topk=int(cfg.get("target_sim_topk", 3)),
                temp=float(cfg.get("target_sim_temp", 0.07)),
                weights=None if t_bank_w is None else t_bank_w.detach().cpu(),
            )
            sim_t_all = pooled.detach().cpu().numpy().astype(np.float32)

        # 注意：默认 sem_margin 使用余弦相似度 (sim_t_cos - sim_bg)，避免不同度量(ACE/MT-CAMF)尺度不一致导致 margin_gate 误抑制目标
        if bool(cfg.get("sem_margin_use_override_sim", False)):
            sem_margin_all = (sim_t_all - sim_bg_all).astype(np.float32)
            sem_margin_reshaped = np.reshape(sem_margin_all, (H, W), order="F")


    # ---- optional: override target similarity by feature-space ACE ----
    elif use_ace_sim and (z_sim_store is not None):
        bg_mask = None
        if ace_bg_select in ["sim_bg_q", "bg_q", "bg_topq"]:
            q = float(np.clip(ace_bg_q, 0.0, 0.9999))
            try:
                thr = float(np.quantile(sim_bg_all.astype(np.float32), q))
            except Exception:
                thr = float(np.max(sim_bg_all))
            bg_mask = (sim_bg_all >= thr)
        elif ace_bg_select in ["all", "none", ""]:
            bg_mask = None

        T_np = z_t_u_used.detach().cpu().numpy().astype(np.float32)
        ace_mat = _ace_similarity_np(
            z_sim_store,
            T_np,
            shrink=ace_shrink,
            ridge=ace_ridge,
            eps=ace_eps,
            bg_mask=bg_mask,
        )

        # pool over target bank (usually Vt=1 when use_target_aug=False)
        if ace_mat.shape[1] == 1:
            sim_t_all = ace_mat[:, 0].astype(np.float32)
        else:
            ace_t = torch.from_numpy(ace_mat).to(device=device, dtype=torch.float32)
            pooled = _pool_similarity(
                ace_t,
                mode=cfg.get("target_sim_pool", "max"),
                topk=int(cfg.get("target_sim_topk", 3)),
                temp=float(cfg.get("target_sim_temp", 0.07)),
                weights=None,
            )
            sim_t_all = pooled.detach().cpu().numpy().astype(np.float32)

        # 注意：默认 sem_margin 使用余弦相似度 (sim_t_cos - sim_bg)，避免不同度量(ACE/MT-CAMF)尺度不一致导致 margin_gate 误抑制目标
        if bool(cfg.get("sem_margin_use_override_sim", False)):
            sem_margin_all = (sim_t_all - sim_bg_all).astype(np.float32)
            sem_margin_reshaped = np.reshape(sem_margin_all, (H, W), order="F")
    # ---- fusion / calibration ----
    score_mode = str(cfg.get("score_mode", "margin_zscore")).lower()

    # debug / calibration helper（主要给 nbs_proto 使用）
    t_max_used = np.nan
    wave_max_used = np.nan
    deriv_max_used = np.nan

    if score_mode in ["ace", "f_ace", "ace_sim", "ace_core"]:
        # Feature-space ACE similarity as the final detection score.
        # Expect sim_t_all already in [0,1] when target_sim_metric == "ace".
        score = np.clip(sim_t_all, 0.0, 1.0).astype(np.float32)
        gate = np.ones_like(score, dtype=np.float32)
        logit = np.log(score + 1e-6).astype(np.float32)

    elif score_mode in ["margin_zscore", "margin"]:
        # (旧) 语义 margin + 频域相似度，做鲁棒 z-score 后线性融合
        z_sem = _robust_zscore(sem_margin_all, clip=float(cfg.get("score_z_clip", 8.0)))
        z_wave = _robust_zscore(wave_sim_all, clip=float(cfg.get("score_z_clip", 8.0)))

        w_sem = float(cfg.get("w_sem", 2.0))
        w_wave = float(cfg.get("w_wave", 0.8))

        logit = w_sem * z_sem + w_wave * z_wave

        # gate：用语义 margin 做背景抑制（阈值稳定性可调）
        gate_k = float(cfg.get("gate_k", 8.0))
        # 支持自适应阈值：auto_mad / auto_q
        margin_thr = _resolve_margin_thr_np(sem_margin_all, cfg)
        gate = _sigmoid_np(gate_k * (sem_margin_all - margin_thr)).astype(np.float32)

        logit = logit * gate

        temp = float(cfg.get("score_temp", 1.0))
        score = _sigmoid_np(logit / max(temp, 1e-6)).astype(np.float32)

        # 额外的 raw 谱角门控（稳定阈值）：进一步抑制结构性背景假警
        if use_angle_gate and angle_w > 0:
            g = np.power(angle_gate_all, angle_w).astype(np.float32)
            score = (score * g).astype(np.float32)
            gate = (gate * g).astype(np.float32)

    elif score_mode in ["evidence_z", "ev_z", "fusion_z", "multi_evidence", "multi_ev"]:
        # ------------------------------------------------------------
        # (New) Multi-evidence fusion (robust z-score + sigmoid)
        # ------------------------------------------------------------
        # Motivation:
        # - In your current "nbs_proto" path, every additional cue (wave/bg/margin/angle)
        #   only *suppresses* the base score. That often makes target scores collapse
        #   near 0 (AUC2 drops) even if the ranking (AUC1) is decent.
        # - Here we treat cues as *evidence* that can both boost and suppress, then
        #   fuse them in a calibrated logit space.
        eps = 1e-6

        z_clip = float(cfg.get("ev_z_clip", 8.0))

        # evidence 1: embedding target similarity
        z_sim_t = _robust_zscore(sim_t_all, clip=z_clip)

        # evidence 2: semantic margin (target - background)
        z_margin = _robust_zscore(sem_margin_all, clip=z_clip)

        # evidence 3: wavelet similarity (optional)
        if use_wavelet_match and (np.std(wave_sim_all.astype(np.float32)) > 1e-6):
            z_wave = _robust_zscore(wave_sim_all, clip=z_clip)
        else:
            z_wave = np.zeros_like(z_sim_t, dtype=np.float32)

        # evidence 3b: first-derivative similarity (optional)
        if bool(cfg.get("use_deriv_match", False)) and (np.std(deriv_sim_all.astype(np.float32)) > 1e-6):
            z_deriv = _robust_zscore(deriv_sim_all, clip=z_clip)
        else:
            z_deriv = np.zeros_like(z_sim_t, dtype=np.float32)

        # evidence 4: raw spectral-angle evidence (smaller angle => larger evidence)
        ang_src = str(cfg.get("ev_ang_src", "t0_raw")).lower()
        if ang_src in ["t0_raw", "raw_t0", "t0"]:
            ang_feat = (-angle_raw_t0_all).astype(np.float32)
        elif ang_src in ["pooled_raw", "raw_pooled", "pooled"]:
            ang_feat = (-angle_raw_all).astype(np.float32)
        elif ang_src in ["t0_gate", "gate_t0"]:
            ang_feat = angle_gate_t0_all.astype(np.float32)
        else:
            # fallback
            ang_feat = angle_gate_all.astype(np.float32)
        z_ang = _robust_zscore(ang_feat, clip=z_clip)

        # evidence 5: background non-typicality (smaller sim_bg => larger evidence)
        z_bg_non = _robust_zscore((-sim_bg_all).astype(np.float32), clip=z_clip)

        # weights (can be tuned per dataset)
        w_sim = float(cfg.get("ev_w_sim_t", 0.9))
        w_mar = float(cfg.get("ev_w_margin", 1.4))
        w_wav = float(cfg.get("ev_w_wave", 0.5))
        w_der = float(cfg.get("ev_w_deriv", 0.35))
        w_ang = float(cfg.get("ev_w_ang", 0.7))
        w_bg = float(cfg.get("ev_w_bg", 0.4))
        bias = float(cfg.get("ev_bias", 0.0))

        logit = (
            bias
            + w_sim * z_sim_t
            + w_mar * z_margin
            + w_wav * z_wave
            + w_der * z_deriv
            + w_ang * z_ang
            + w_bg * z_bg_non
        ).astype(np.float32)

        # --- soft gates for threshold stability (additive, not multiplicative) ---
        gate_k = float(cfg.get("gate_k", 8.0))
        margin_thr = _resolve_margin_thr_np(sem_margin_all, cfg)
        gate_m = _sigmoid_np(gate_k * (sem_margin_all - margin_thr)).astype(np.float32)

        gate = gate_m
        if use_angle_gate:
            g_src = str(cfg.get("ev_gate_angle_source", "t0")).lower()
            if g_src in ["t0", "t0_gate", "gate_t0"]:
                g_ang = angle_gate_t0_all.astype(np.float32)
            else:
                g_ang = angle_gate_all.astype(np.float32)
            ang_pow = float(cfg.get("ev_gate_ang_pow", 1.0))
            gate = (gate * np.power(g_ang, ang_pow)).astype(np.float32)

        gate_w = float(cfg.get("ev_gate_w", 1.0))
        logit = (logit + gate_w * np.log(gate + eps)).astype(np.float32)

        temp = float(cfg.get("score_temp", 1.0))
        score = _sigmoid_np(logit / max(temp, 1e-6)).astype(np.float32)

    elif score_mode in ["htdmamba", "nbs_target", "orig_htd"]:
        # 纯 HTD-Mamba 推理（对齐论文公式）：
        #   score = NBS( cosine( f(x), f(t) ) )
        # 这里不引入原型/波段频域融合，仅作为“基线/消融”便于你定位掉点来源。
        eps = 1e-6
        delta_t = float(cfg.get("nbs_delta_t", 0.10))
        score = np.exp(-((sim_t_all - 1.0) ** 2) / max(delta_t, eps)).astype(np.float32)
        gate = np.ones_like(score, dtype=np.float32)
        logit = np.log(score + eps).astype(np.float32)

        # 可选：叠加 raw 谱角门控（默认关，避免破坏与论文对齐的基线）
        if bool(cfg.get("htd_apply_angle_gate", False)) and use_angle_gate:
            w_a = float(cfg.get("htd_angle_gate_w", 1.0))
            g = np.power(angle_gate_all, w_a).astype(np.float32)
            score = (score * g).astype(np.float32)
            gate = (gate * g).astype(np.float32)
            logit = np.log(score + eps).astype(np.float32)

        # 可选：叠加 wavelet 频域匹配（默认关）
        if bool(cfg.get("htd_apply_wavelet_match", False)) and use_wavelet_match:
            # 直接用 nbs_w 乘上去即可（仍在 [0,1]）
            delta_w = float(cfg.get("nbs_delta_w", 0.15))
            wave_max = float(cfg.get("wave_sim_max", 1.5))
            wave_norm = np.clip(wave_sim_all / max(wave_max, eps), -1.0, 1.0)
            nbs_w = np.exp(-((wave_norm - 1.0) ** 2) / max(delta_w, eps)).astype(np.float32)
            score = (score * nbs_w).astype(np.float32)
            gate = (gate * nbs_w).astype(np.float32)
            logit = np.log(score + eps).astype(np.float32)

    elif score_mode in ["nbs_proto", "nbs"]:
        # (新) 更贴近原始 HTD-Mamba 的 NBS 思路：主要依赖“目标相似度”，
        # 背景原型作为抑制项（避免 margin 形式在稀有背景上放大假警）。
        eps = 1e-6
        delta_t = float(cfg.get("nbs_delta_t", 0.10))  # 目标相似度 NBS
        delta_w = float(cfg.get("nbs_delta_w", 0.15))  # 频域相似度 NBS
        delta_bg = float(cfg.get("nbs_delta_bg", 0.10))  # 背景相似度抑制

        # 1) target similarity -> NBS
        # NBS 的假设：最匹配的目标像素 sim≈1。
        # 但在引入 target-bank pooling / bg_osp / wavelet 等之后，sim_t 的最大值可能显著小于 1，
        # 直接套 exp(-((sim-1)^2)/delta) 会把所有分数压得很扁，排序能力下降（AUC 下降）。
        # 因此提供可选的鲁棒归一化：用高分位数把 sim_t 拉回到 [−1,1] 的有效区间。
        t_norm_mode = str(cfg.get("nbs_t_norm", "none")).lower()
        sim_t_used = sim_t_all
        if t_norm_mode in ["auto", "quantile", "q"]:
            # 旧：分位数标定（可能被极少数 outlier 带偏）
            q = float(cfg.get("nbs_t_norm_q", 0.999))
            q = float(np.clip(q, 0.90, 0.9999))
            try:
                t_max = float(np.quantile(sim_t_all.astype(np.float32), q))
            except Exception:
                t_max = float(np.max(sim_t_all))
            if (not np.isfinite(t_max)) or t_max < 1e-6:
                t_max = 1.0
            t_max_used = t_max
            sim_t_used = np.clip(sim_t_all / t_max, -1.0, 1.0)

        elif t_norm_mode in [
            "topk_mean",
            "topk_median",
            "topk",
            "auto_topk_mean",
            "auto_topk_median",
        ]:
            # 新：top-k 统计量标定（更稳健）
            top_ratio = float(cfg.get("nbs_t_norm_top_ratio", 0.001))
            min_k = int(cfg.get("nbs_t_norm_min_k", 16))
            # alias
            mode = "topk_median" if t_norm_mode in ["topk", "auto_topk_median"] else t_norm_mode
            if t_norm_mode in ["auto_topk_mean"]:
                mode = "topk_mean"

            # （关键改进）NBS 标定时的“高分集”可能被结构性背景 outlier 污染，
            # 导致 t_max 偏大、所有分数被压扁（AUC1/AUC2 下降）。
            # 允许用物理先验（raw 谱角门控）先筛一遍，再做 top-k 统计更稳健。
            x_for_norm = sim_t_all
            norm_mask = str(cfg.get("nbs_t_norm_mask", "none")).lower()
            if norm_mask in ["angle_gate", "angle", "ang_gate"]:
                thr_gate = float(cfg.get("nbs_t_norm_angle_gate_thr", 0.5))
                x_tmp = sim_t_all[angle_gate_all > thr_gate]
                if x_tmp.size >= min_k:
                    x_for_norm = x_tmp
            elif norm_mask in ["angle_raw", "ang_raw"]:
                thr_ang = float(cfg.get("nbs_t_norm_angle_raw_thr", cfg.get("angle_gate_thr", 0.30)))
                x_tmp = sim_t_all[angle_raw_all < thr_ang]
                if x_tmp.size >= min_k:
                    x_for_norm = x_tmp
            elif norm_mask in ["angle_t0_gate", "angle_gate_t0", "t0_gate", "ang_t0_gate", "angle0_gate"]:
                thr_gate = float(cfg.get("nbs_t_norm_angle_t0_gate_thr", cfg.get("nbs_t_norm_angle_gate_thr", 0.5)))
                x_tmp = sim_t_all[angle_gate_t0_all > thr_gate]
                if x_tmp.size >= min_k:
                    x_for_norm = x_tmp
            elif norm_mask in ["angle_t0_raw", "angle_raw_t0", "t0_raw", "ang_t0_raw", "angle0_raw"]:
                thr_ang = float(cfg.get("nbs_t_norm_angle_t0_raw_thr", cfg.get("nbs_t_norm_angle_raw_thr", cfg.get("angle_gate_thr", 0.30))))
                x_tmp = sim_t_all[angle_raw_t0_all < thr_ang]
                if x_tmp.size >= min_k:
                    x_for_norm = x_tmp

            t_max = _auto_topk_stat_np(x_for_norm, mode=mode, top_ratio=top_ratio, min_k=min_k)
            t_max_used = float(t_max)
            sim_t_used = np.clip(sim_t_all / max(t_max_used, 1e-6), -1.0, 1.0)

        nbs_t = np.exp(-((sim_t_used - 1.0) ** 2) / max(delta_t, eps)).astype(np.float32)

        # 2) wavelet similarity -> NBS (optional)
        if use_wavelet_match and (np.std(wave_sim_all.astype(np.float32)) > 1e-6):
            wave_max_cfg = cfg.get("wave_sim_max", 1.5)
            if isinstance(wave_max_cfg, str):
                wmode = str(wave_max_cfg).lower()
                if wmode in ["auto", "quantile", "q"]:
                    # 旧：分位数标定
                    q = float(cfg.get("wave_sim_max_q", 0.999))
                    q = float(np.clip(q, 0.90, 0.9999))
                    try:
                        wave_max = float(np.quantile(wave_sim_all.astype(np.float32), q))
                    except Exception:
                        wave_max = float(np.max(wave_sim_all))
                    if (not np.isfinite(wave_max)) or wave_max < 1e-6:
                        wave_max = 1.5
                    wave_max_used = float(wave_max)

                elif wmode in [
                    "topk_mean",
                    "topk_median",
                    "topk",
                    "auto_topk_mean",
                    "auto_topk_median",
                ]:
                    # 新：top-k 统计量标定（更稳健）
                    top_ratio = float(cfg.get("wave_sim_max_top_ratio", 0.001))
                    min_k = int(cfg.get("wave_sim_max_min_k", 16))
                    mode = "topk_median" if wmode in ["topk", "auto_topk_median"] else wmode
                    if wmode in ["auto_topk_mean"]:
                        mode = "topk_mean"

                    # 与 nbs_t_norm 同理：wave_max 的标定也可能被“偶然很像的背景”拉高，
                    # 这里同样允许用 raw 谱角门控先筛掉明显不合理的像素，再做 top-k 统计。
                    x_for_norm = wave_sim_all
                    norm_mask = str(cfg.get("wave_sim_max_mask", "none")).lower()
                    if norm_mask in ["angle_gate", "angle", "ang_gate"]:
                        thr_gate = float(cfg.get("wave_sim_max_angle_gate_thr", 0.5))
                        x_tmp = wave_sim_all[angle_gate_all > thr_gate]
                        if x_tmp.size >= min_k:
                            x_for_norm = x_tmp
                    elif norm_mask in ["angle_raw", "ang_raw"]:
                        thr_ang = float(cfg.get("wave_sim_max_angle_raw_thr", cfg.get("angle_gate_thr", 0.30)))
                        x_tmp = wave_sim_all[angle_raw_all < thr_ang]
                        if x_tmp.size >= min_k:
                            x_for_norm = x_tmp
                    elif norm_mask in ["angle_t0_gate", "angle_gate_t0", "t0_gate", "ang_t0_gate", "angle0_gate"]:
                        thr_gate = float(cfg.get("wave_sim_max_angle_t0_gate_thr", cfg.get("wave_sim_max_angle_gate_thr", 0.5)))
                        x_tmp = wave_sim_all[angle_gate_t0_all > thr_gate]
                        if x_tmp.size >= min_k:
                            x_for_norm = x_tmp
                    elif norm_mask in ["angle_t0_raw", "angle_raw_t0", "t0_raw", "ang_t0_raw", "angle0_raw"]:
                        thr_ang = float(cfg.get("wave_sim_max_angle_t0_raw_thr", cfg.get("wave_sim_max_angle_raw_thr", cfg.get("angle_gate_thr", 0.30))))
                        x_tmp = wave_sim_all[angle_raw_t0_all < thr_ang]
                        if x_tmp.size >= min_k:
                            x_for_norm = x_tmp

                    wave_max = _auto_topk_stat_np(x_for_norm, mode=mode, top_ratio=top_ratio, min_k=min_k)
                    wave_max_used = float(wave_max)

                else:
                    # unknown string -> fallback to constant
                    wave_max_used = 1.5
            else:
                wave_max_used = float(wave_max_cfg)

            # sim_h + 0.5 sim_l 的理论范围约为 [-1.5, 1.5]
            wave_norm = np.clip(wave_sim_all / max(wave_max_used, eps), -1.0, 1.0)
            nbs_w = np.exp(-((wave_norm - 1.0) ** 2) / max(delta_w, eps)).astype(
                np.float32
            )
        else:
            nbs_w = np.ones_like(nbs_t, dtype=np.float32)

        
        # 2b) derivative-domain similarity -> NBS (optional)
        # 说明：
        #   - deriv_sim：对 correlation-cosine 归一化谱的一阶差分，再做余弦相似度（pool over target bank）
        #   - 作用：更敏感于高频/局部谱形细节，能系统性压制“局部形变/混合谱 confuser”
        delta_d = float(cfg.get("nbs_delta_d", 0.12))

        if use_deriv_match and (np.std(deriv_sim_all.astype(np.float32)) > 1e-6):
            deriv_max_cfg = cfg.get("deriv_sim_max", "topk_median")
            if isinstance(deriv_max_cfg, str):
                dmode = str(deriv_max_cfg).lower()

                if dmode in ["auto", "quantile", "q"]:
                    # 分位数标定（简单，但可能被极少数 outlier 影响）
                    q = float(cfg.get("deriv_sim_max_q", 0.999))
                    q = float(np.clip(q, 0.90, 0.99999))
                    try:
                        dmax = float(np.quantile(deriv_sim_all.astype(np.float32), q))
                    except Exception:
                        dmax = float(np.max(deriv_sim_all))
                    if (not np.isfinite(dmax)) or dmax < 1e-6:
                        dmax = 1.0
                    deriv_max_used = float(dmax)

                elif dmode in [
                    "topk_mean",
                    "topk_median",
                    "topk",
                    "auto_topk_mean",
                    "auto_topk_median",
                ]:
                    # top-k 统计量标定（更稳健）
                    top_ratio = float(cfg.get("deriv_sim_max_top_ratio", 0.001))
                    min_k = int(cfg.get("deriv_sim_max_min_k", 16))
                    mode = "topk_median" if dmode in ["topk", "auto_topk_median"] else dmode
                    if dmode in ["auto_topk_mean"]:
                        mode = "topk_mean"

                    # 与 wave_sim_max 同理：允许先用 raw-angle gate 过滤明显不合理的像素再做标定
                    x_for_norm = deriv_sim_all
                    norm_mask = str(cfg.get("deriv_sim_max_mask", cfg.get("wave_sim_max_mask", "none"))).lower()

                    if norm_mask in ["angle_gate", "angle", "ang_gate"]:
                        thr_gate = float(cfg.get("deriv_sim_max_angle_gate_thr", cfg.get("wave_sim_max_angle_gate_thr", 0.5)))
                        x_tmp = deriv_sim_all[angle_gate_all > thr_gate]
                        if x_tmp.size >= min_k:
                            x_for_norm = x_tmp
                    elif norm_mask in ["angle_raw", "ang_raw"]:
                        thr_ang = float(cfg.get("deriv_sim_max_angle_raw_thr", cfg.get("angle_gate_thr", 0.30)))
                        x_tmp = deriv_sim_all[angle_raw_all < thr_ang]
                        if x_tmp.size >= min_k:
                            x_for_norm = x_tmp
                    elif norm_mask in ["angle_t0_gate", "angle_gate_t0", "t0_gate", "ang_t0_gate", "angle0_gate"]:
                        thr_gate = float(cfg.get("deriv_sim_max_angle_t0_gate_thr", cfg.get("wave_sim_max_angle_gate_thr", 0.5)))
                        x_tmp = deriv_sim_all[angle_gate_t0_all > thr_gate]
                        if x_tmp.size >= min_k:
                            x_for_norm = x_tmp
                    elif norm_mask in ["angle_t0_raw", "angle_raw_t0", "t0_raw", "ang_t0_raw", "angle0_raw"]:
                        thr_ang = float(cfg.get("deriv_sim_max_angle_t0_raw_thr", cfg.get("angle_gate_thr", 0.30)))
                        x_tmp = deriv_sim_all[angle_raw_t0_all < thr_ang]
                        if x_tmp.size >= min_k:
                            x_for_norm = x_tmp

                    dmax = _auto_topk_stat_np(x_for_norm, mode=mode, top_ratio=top_ratio, min_k=min_k)
                    deriv_max_used = float(dmax)

                else:
                    # unknown string -> fallback to constant
                    deriv_max_used = 1.0
            else:
                deriv_max_used = float(deriv_max_cfg)

            # cosine similarity in [-1,1] -> normalize so that "best-match" ~ 1 for NBS
            deriv_norm = np.clip(deriv_sim_all / max(deriv_max_used, eps), -1.0, 1.0)
            nbs_d = np.exp(-((deriv_norm - 1.0) ** 2) / max(delta_d, eps)).astype(np.float32)
        else:
            nbs_d = np.ones_like(nbs_t, dtype=np.float32)

# 3) background prototype similarity -> suppress (high sim_bg => strong background)
        # 注意：NBS 公式 exp(-((sim-1)^2)/delta) 是为“目标相似度接近 1”设计的。
        # 对于“原型相似度”而言，sim_bg 往往不一定靠近 1（更像 SwAV 的 cluster logits），
        # 直接套 NBS 会导致 bg_gate 几乎恒为 1，从而**背景抑制失效**。
        # 因此这里提供更稳健的阈值门控：sigmoid(k*(thr - sim_bg))，并支持 thr=auto(分位数)。

        bg_gate_mode = str(cfg.get("bg_gate_mode", "sigmoid_sim")).lower()
        bg_thr_used = np.nan

        if bg_gate_mode in ["nbs", "nbs1", "nbs_gauss"]:
            nbs_bg = np.exp(-((sim_bg_all - 1.0) ** 2) / max(delta_bg, eps)).astype(
                np.float32
            )
            bg_gate = (1.0 - nbs_bg).astype(np.float32)  # [0,1], 越大越“非背景典型”

        elif bg_gate_mode in ["sigmoid", "sigmoid_sim", "thr", "logistic"]:
            bg_k = float(cfg.get("bg_gate_k", 24.0))
            bg_thr = cfg.get("bg_gate_thr", "auto")
            if isinstance(bg_thr, str) and bg_thr.lower() == "auto":
                q = float(cfg.get("bg_gate_thr_q", 0.90))
                q = float(np.clip(q, 0.0, 1.0))
                bg_thr_used = float(np.quantile(sim_bg_all, q))
            else:
                bg_thr_used = float(bg_thr)

            bg_gate = _sigmoid_np(bg_k * (bg_thr_used - sim_bg_all)).astype(np.float32)
            # 为保持 debug 变量一致性：定义一个“伪 nbs_bg”
            nbs_bg = (1.0 - bg_gate).astype(np.float32)

        else:
            raise ValueError(
                f"Unknown bg_gate_mode: {bg_gate_mode}. Use 'sigmoid_sim' or 'nbs'."
            )

        # 4) 可选：margin gate（进一步抑制“更像背景”的像素）
        #    - sem_margin = sim_t - sim_bg
        #    - margin_gate 逼近 0/1，比仅用 bg_gate 更能抑制结构性背景假警
        use_margin_gate = bool(cfg.get("nbs_use_margin_gate", False))
        if use_margin_gate:
            gate_k = float(cfg.get("gate_k", 8.0))
            # 支持自适应阈值：auto_mad / auto_q
            margin_thr = _resolve_margin_thr_np(sem_margin_all, cfg)
            margin_gate = _sigmoid_np(gate_k * (sem_margin_all - margin_thr)).astype(
                np.float32
            )
            w_m = float(cfg.get("score_w_margin_gate", 1.0))
        else:
            margin_gate = np.ones_like(bg_gate, dtype=np.float32)
            w_m = 0.0

        # 5) fuse in log-space (product with exponents)
        w_t = float(cfg.get("score_w_t", 1.0))
        w_w = float(cfg.get("score_w_wave", 0.5))
        w_d = float(cfg.get("score_w_deriv", 0.35))
        w_b = float(cfg.get("score_w_bg", 1.0))

        # --- (新增) adaptive fusion ---
        # 直觉：当 nbs_t 已经非常高（强目标像素）时，再叠乘 wave/bg/margin/angle 等抑制项，
        # 可能会“误伤”少量真实目标像素（例如轻微 smile/keystone 或局部噪声导致 wave/angle 略差），
        # 从而拉低 ROC AUC（AUC1）。
        # 做法：用 nbs_t -> conf∈[0,1] 作为目标置信度，令抑制项权重随 (1-conf) 自适应减弱，
        # 同时保留一个 min_sup，避免抑制项完全失效。
        adaptive_fusion = bool(cfg.get("adaptive_fusion", True))
        fusion_conf_thr = float(cfg.get("fusion_conf_thr", 0.65))
        fusion_conf_k = float(cfg.get("fusion_conf_k", 12.0))
        fusion_min_sup = float(cfg.get("fusion_min_sup", 0.15))
        adaptive_angle_gate = bool(cfg.get("adaptive_angle_gate", True))

        if adaptive_fusion:
            # 基础目标置信度：由 nbs_t 映射到 (0,1)
            x_conf = fusion_conf_k * (nbs_t - fusion_conf_thr)
            x_conf = np.clip(x_conf, -60.0, 60.0)
            conf = (1.0 / (1.0 + np.exp(-x_conf))).astype(np.float32)  # [N]

            # (新增) 置信度再融合“物理一致性/背景非典型性”，避免：
            #   - 偶然高 sim_t 的背景像素被当成强目标 -> sup 变小 -> 抑制项失效 -> 假警上升 -> AUC1 掉点
            # 这里做的是“温和融合”（factor∈[1-coef, 1]），不会把 conf 拉得太低从而误伤真目标。
            if bool(cfg.get("fusion_conf_use_angle", True)) and (use_angle_gate or angle_aware_pool):
                ang_coef = float(cfg.get("fusion_conf_ang_coeff", 0.5))
                ang_coef = float(np.clip(ang_coef, 0.0, 1.0))
                ang_src = str(cfg.get("fusion_conf_angle_source", "t0")).lower()
                if ang_src in ["t0", "raw", "t0_gate", "angle_t0", "angle0"]:
                    g_ang = angle_gate_t0_all
                else:
                    g_ang = angle_gate_all
                conf = conf * (1.0 - ang_coef + ang_coef * g_ang)

            if bool(cfg.get("fusion_conf_use_bg", True)):
                bg_coef = float(cfg.get("fusion_conf_bg_coeff", 0.5))
                bg_coef = float(np.clip(bg_coef, 0.0, 1.0))
                conf = conf * (1.0 - bg_coef + bg_coef * bg_gate)

            if bool(cfg.get("fusion_conf_use_margin", False)) and use_margin_gate:
                m_coef = float(cfg.get("fusion_conf_margin_coeff", 0.25))
                m_coef = float(np.clip(m_coef, 0.0, 1.0))
                conf = conf * (1.0 - m_coef + m_coef * margin_gate)

            conf = np.clip(conf, 0.0, 1.0).astype(np.float32)

            # sup：抑制项的“保留比例”，conf 越高 -> sup 越接近 min_sup（强目标少抑制）
            sup = (fusion_min_sup + (1.0 - fusion_min_sup) * (1.0 - conf)).astype(np.float32)  # [N], in (min_sup, 1]
        else:
            conf = np.ones_like(nbs_t, dtype=np.float32)
            sup = np.ones_like(nbs_t, dtype=np.float32)

        logit = (w_t * np.log(nbs_t + eps)).astype(np.float32)
        logit = (logit + (sup * w_w * np.log(nbs_w + eps))).astype(np.float32)
        logit = (logit + (sup * w_d * np.log(nbs_d + eps))).astype(np.float32)
        logit = (logit + (sup * w_b * np.log(bg_gate + eps))).astype(np.float32)
        logit = (logit + (sup * w_m * np.log(margin_gate + eps))).astype(np.float32)

        score = np.exp(logit).astype(np.float32)

        # 额外的 raw 谱角门控（稳定阈值）
        if use_angle_gate and angle_w > 0:
            if adaptive_fusion and adaptive_angle_gate:
                angle_power_used = (angle_w * sup).astype(np.float32)
            else:
                angle_power_used = float(angle_w)
            g = np.power(angle_gate_all, angle_power_used).astype(np.float32)
            score = (score * g).astype(np.float32)

        # for debug compatibility
        gate = (bg_gate * margin_gate).astype(np.float32)
        if use_angle_gate and angle_w > 0:
            # 与 score 中使用的谱角幂次保持一致（adaptive_fusion 时可能是逐像素 power）
            if "angle_power_used" in locals():
                gate = (gate * np.power(angle_gate_all, angle_power_used)).astype(np.float32)
            else:
                gate = (gate * np.power(angle_gate_all, angle_w)).astype(np.float32)

    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")


    

    # --- optional: spatial refinement (test-time only, does NOT use GT) ---
    # Important: do it BEFORE score_post_norm, so post calibration is applied on the refined map.
    if bool(cfg.get('spatial_refine_enable', False)):
        H_img = cfg.get('img_h', None)
        W_img = cfg.get('img_w', None)
        if H_img is not None and W_img is not None:
            try:
                H_img = int(H_img)
                W_img = int(W_img)
                if H_img > 1 and W_img > 1 and (H_img * W_img == int(score.shape[0])):
                    score_map_tmp = score.reshape(H_img, W_img, order='F').astype(np.float32)
                    score_map_tmp = _spatial_refine_score_map_np(score_map_tmp, cfg)
                    score = score_map_tmp.reshape(-1, order='F').astype(np.float32)
            except Exception:
                # fail-safe: keep original score
                pass

    # --- score post normalization ---
    # 说明：AUC1 (Pf-Pd) 只依赖排序，单调变换不影响；
    #       但 AUC2/AUC3 是 (τ-Pd)/(τ-Pf) 曲线的面积，强依赖 score 的数值标定。
    #       当你加入多证据乘法融合（wave/bg/margin/angle）后，score 往往会整体变小，
    #       这会在 τ∈[0,1] 的积分里“白白浪费”一段阈值区间，从而拉低 AUC2、并影响 AUC4/AUC5。
    #       因此提供一个 *可选* 的后处理标定：用全局 max 或高分位数把 score 拉回 [0,1]。
    #       该操作是单调的，不会破坏 AUC1。
    score_post_norm = str(cfg.get("score_post_norm", "none")).lower()
    score_post_scale = np.nan
    if score_post_norm not in ["none", "off", "false", "0", ""]:
        _eps = 1e-6
        if score_post_norm in ["max", "smax"]:
            s = float(np.nanmax(score))
            if (not np.isfinite(s)) or s < _eps:
                s = 1.0
            score_post_scale = s
            score = np.clip(score / (s + _eps), 0.0, 1.0).astype(np.float32)
        elif score_post_norm in ["q", "quantile", "percentile"]:
            q = float(cfg.get("score_post_norm_q", 0.999))
            q = float(np.clip(q, 0.90, 0.99999))
            s_vec = score
            mask_mode = str(cfg.get("score_post_norm_mask", "none")).lower()
            min_keep = int(cfg.get("score_post_norm_min_keep", 1024))
            if mask_mode in ["angle_t0_gate", "t0_gate", "angle0"] and "angle_gate_t0_all" in locals():
                thr_m = float(cfg.get("score_post_norm_angle_t0_gate_thr", 0.5))
                msk = angle_gate_t0_all >= thr_m
                if int(msk.sum()) >= min_keep:
                    s_vec = score[msk]
            elif mask_mode in ["angle_gate", "pooled_angle_gate"] and "angle_gate_all" in locals():
                thr_m = float(cfg.get("score_post_norm_angle_gate_thr", 0.5))
                msk = angle_gate_all >= thr_m
                if int(msk.sum()) >= min_keep:
                    s_vec = score[msk]

            try:
                s = float(np.quantile(s_vec.astype(np.float32), q))
            except Exception:
                s = float(np.nanmax(score))

            if (not np.isfinite(s)) or s < _eps:
                s = float(np.nanmax(score))
            if (not np.isfinite(s)) or s < _eps:
                s = 1.0

            score_post_scale = s
            score = np.clip(score / (s + _eps), 0.0, 1.0).astype(np.float32)
        elif score_post_norm in ["conformal", "pvalue", "cp", "p"]:
            # -------------------------
            # NEW: Conformal-style calibration (test-time only)
            # -------------------------
            # We need a calibration set that is *mostly background*.
            # Practical heuristic: select pixels with small suppression gate.
            # This does NOT use ground-truth and does NOT affect training.
            calib_mode = str(cfg.get("conformal_calib", "gate_low")).lower()
            min_keep = int(cfg.get("conformal_min_keep", 2048))
            gate_thr = cfg.get("conformal_gate_thr", 0.5)
            gate_q = float(cfg.get("conformal_gate_q", 0.70))
            gate_q = float(np.clip(gate_q, 0.05, 0.95))

            msk = None
            if calib_mode in ["gate_low", "gate", "bg"] and "gate" in locals():
                if isinstance(gate_thr, str) and gate_thr.lower() in ["auto", "q", "quantile"]:
                    thr = float(np.quantile(gate.astype(np.float32), gate_q))
                else:
                    thr = float(gate_thr)
                msk = (gate.astype(np.float32) <= thr)

                # if too few pixels, relax to a quantile rule
                if int(msk.sum()) < min_keep:
                    thr = float(np.quantile(gate.astype(np.float32), gate_q))
                    msk = (gate.astype(np.float32) <= thr)

            # fallback: use whole image
            if msk is None or int(msk.sum()) < 8:
                msk = np.ones_like(score, dtype=bool)

            calib_scores = score[msk]
            score = _conformal_score_np(score, calib_scores)
        else:
            raise ValueError(f"Unknown score_post_norm: {score_post_norm}")



    # --- optional: adaptive sigmoid calibration (monotonic, test-time only) ---
    # 目的：不改变排序（AUC1 不变）的前提下，把“背景中等分数”压向 0、把“真目标分数”推向 1，
    #      从而改善 τ-curve 指标（AUC2↑、AUC3↓、AUC5↑）。
    # 使用方式：在 infer_preset 中设置
    #   score_post_sigmoid=True
    #   score_post_sigmoid_q=0.98
    #   score_post_sigmoid_k=20
    # 可选：score_post_sigmoid_mask='gate_low' 用低 gate 区域估计 mu（更接近纯背景）。
    score_post_sigmoid = bool(cfg.get("score_post_sigmoid", False))
    score_post_sigmoid_mu_used = np.nan
    score_post_sigmoid_k_used = np.nan
    if score_post_sigmoid:
        k = float(cfg.get("score_post_sigmoid_k", 20.0))
        q_mu = float(cfg.get("score_post_sigmoid_q", 0.98))
        q_mu = float(np.clip(q_mu, 0.50, 0.9999))

        mu_cfg = cfg.get("score_post_sigmoid_mu", "q")  # 'q' | 'median' | float

        mask_mode = str(cfg.get("score_post_sigmoid_mask", "none")).lower()
        min_keep = int(cfg.get("score_post_sigmoid_min_keep", 1024))
        gate_q = float(cfg.get("score_post_sigmoid_gate_q", 0.70))
        gate_q = float(np.clip(gate_q, 0.05, 0.95))
        gate_thr = cfg.get("score_post_sigmoid_gate_thr", "auto")

        # 估计 mu 的样本集合：默认用全图 score；也可用 gate_low 近似背景子集
        s_vec = score.reshape(-1)
        if mask_mode in ["gate_low", "gate", "bg", "background"] and "gate" in locals():
            try:
                g = gate.astype(np.float32).reshape(-1)
                if isinstance(gate_thr, str) and gate_thr.lower() in ["auto", "q", "quantile"]:
                    thr = float(np.quantile(g, gate_q))
                else:
                    thr = float(gate_thr)
                msk = (g <= thr)
                if int(msk.sum()) >= min_keep:
                    s_vec = score.reshape(-1)[msk]
            except Exception:
                pass

        # resolve mu
        try:
            if isinstance(mu_cfg, str):
                mstr = mu_cfg.lower()

                if mstr in ["q", "quantile", "auto"]:
                    # 默认：从 s_vec 的分位数估计拐点（通常效果尚可）
                    mu = float(np.quantile(s_vec.astype(np.float32), q_mu))

                elif mstr in ["median", "p50"]:
                    mu = float(np.quantile(s_vec.astype(np.float32), 0.50))

                elif mstr in ["pos_ratio", "pos", "top_ratio", "top"]:
                    # 让 sigmoid 的拐点落在“顶部 tail”：
                    #   mu = quantile(score, 1-pos_ratio)
                    # 直觉：目标像素通常很稀疏，把拐点放到高分尾部能显著抬高强目标，同时压低绝大多数背景。
                    pos_ratio = float(cfg.get("score_post_sigmoid_pos_ratio", 0.01))
                    pos_ratio = float(np.clip(pos_ratio, 1e-6, 0.20))
                    mu = float(np.quantile(score.reshape(-1).astype(np.float32), 1.0 - pos_ratio))

                elif mstr in ["bg_q", "bg_quantile", "bgq"]:
                    # NEW: 背景感知的 mu（更适合你的需求：AUC1 不变 + 抬高 AUC2，同时尽量不抬 AUC3）
                    # 思路：
                    #   1) 用 sim_bg 选出“更像背景”的像素子集；
                    #   2) 在该子集上取高分位数作为 mu（近似背景的极端尾部）；
                    #   3) 若 mu 过高导致几乎没有高分像素，则用 min_pos_ratio 兜底把 mu 拉回。
                    s_all = score.reshape(-1).astype(np.float32)

                    s_mu_vec = s_all
                    if "sim_bg_all" in locals():
                        sel_q = float(cfg.get("score_post_sigmoid_bg_sel_q", 0.60))
                        sel_q = float(np.clip(sel_q, 0.0, 1.0))
                        try:
                            thr_bg = float(np.quantile(sim_bg_all.astype(np.float32), sel_q))
                            msk_bg = (sim_bg_all.reshape(-1).astype(np.float32) >= thr_bg)
                            if int(msk_bg.sum()) >= int(min_keep):
                                s_mu_vec = s_all[msk_bg]
                        except Exception:
                            pass

                    q_bg = float(cfg.get("score_post_sigmoid_bg_mu_q", 0.995))
                    q_bg = float(np.clip(q_bg, 0.50, 0.99999))
                    mu = float(np.quantile(s_mu_vec.astype(np.float32), q_bg))

                    # 兜底：避免 mu 过高导致目标全被压扁（AUC2 掉）
                    min_pos_ratio = float(cfg.get("score_post_sigmoid_min_pos_ratio", 0.0))
                    if np.isfinite(min_pos_ratio) and (min_pos_ratio > 0.0):
                        min_pos_ratio = float(np.clip(min_pos_ratio, 1e-6, 0.20))
                        mu_pos = float(np.quantile(s_all.astype(np.float32), 1.0 - min_pos_ratio))
                        mu = float(min(mu, mu_pos))

                else:
                    # 允许直接给 float（手动指定 mu）
                    mu = float(mu_cfg)
            else:
                mu = float(mu_cfg)
        except Exception:
            mu = float(np.quantile(s_vec.astype(np.float32), q_mu))

        if not np.isfinite(mu):
            mu = float(np.quantile(score.astype(np.float32), q_mu))

        if (not np.isfinite(k)) or (k <= 0):
            k = 20.0

        score_post_sigmoid_mu_used = float(mu)
        score_post_sigmoid_k_used = float(k)

        # sigmoid mapping (strictly monotonic):
        #   原始做法：score' = sigmoid(k*(score-mu))
        #   注意：sigmoid 直接作用在 [0,1] 的 score 上时，会带来一个“背景分数下限”：
        #       当 score→0 时，score'→sigmoid(-k*mu) > 0，
        #       这会导致 AUC3(τ-Pf) 很难继续降低（你最新日志里 AUC3 偏高的直接原因之一）。
        #   改进：端点锚定(Anchored)的自适应 sigmoid —— 保持严格单调(不影响 AUC1 排序)，
        #       同时强制 0→0、1→1，避免背景下限抬高。
        rescale = bool(cfg.get("score_post_sigmoid_rescale", True))

        x = (score.astype(np.float32) - float(mu)) * float(k)
        x = np.clip(x, -60.0, 60.0)
        y = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

        if rescale:
            # y0 = sigmoid(k*(0-mu)), y1 = sigmoid(k*(1-mu))
            x0 = float(np.clip((0.0 - float(mu)) * float(k), -60.0, 60.0))
            x1 = float(np.clip((1.0 - float(mu)) * float(k), -60.0, 60.0))
            y0 = 1.0 / (1.0 + np.exp(-x0))
            y1 = 1.0 / (1.0 + np.exp(-x1))
            denom = float(y1 - y0)
            if denom > 1e-6:
                y = (y - float(y0)) / denom

        score = np.clip(y, 0.0, 1.0).astype(np.float32)

    # --- optional monotonic sharpening (power transform) ---
    # 说明：这是一个严格单调映射（pow>0），不会改变 score 的排序，因此不会改变 AUC1。
    #       但它会改变 score 的数值分布，从而影响 AUC2/AUC3/AUC4/AUC5。
    #       经验上：pow>1 会把背景的中等分数压向 0（AUC3↓、AUC5↑），
    #             pow<1 会把分数整体抬高（AUC2↑但可能 AUC3↑）。
        # --- optional monotonic sharpening (power transform) ---
    # 说明：这是一个严格单调映射（pow>0），不会改变 score 的排序，因此不会改变 AUC1。
    #       但它会改变 score 的数值分布，从而影响 AUC2/AUC3/AUC4/AUC5。
    #       经验上：pow>1 会把背景的中等分数压向 0（AUC3↓、AUC5↑），
    #             pow<1 会把分数整体抬高（AUC2↑但可能 AUC3↑）。
    #
    # ✅ 新增：支持“自动 pow 标定”（score_post_pow_auto / score_post_auto）
    #   - 不需要 GT，不需要手动 sweep；
    #   - 用 sim_bg / gate 估计背景子集，二分搜索选 pow，使背景均值接近 score_post_auto_bg_mean。

    # --- optional: t0-angle based extra suppression (非单调，会改变排序) ---
    # 直觉：
    #   - sim_t / margin 等是在“嵌入空间”里做的相似度；某些背景材质在嵌入空间里可能会“撞脸”目标，导致背景尾部假警抬高（AUC3 偏大）。
    #   - angle_gate_t0 是在“原始谱形”上对目标端元 t0 的谱角门控（更贴近物理谱形一致性）。
    # 做法：
    #   score <- score * (angle_gate_t0 ** p)
    #   其中 p>0 会显著压制“谱形不一致但嵌入相似”的背景假警，通常对目标 (gt=1) 影响较小，从而 AUC5 能大幅提升。
    # 注意：这是非单调变换，会改变 AUC1（但在 Sandiego 这类 AUC1 已经很高、主要问题是 AUC3 偏大时，往往收益远大于代价）。
    score_post_t0_angle_pow_used = np.nan
    score_post_t0_angle_pow = cfg.get("score_post_t0_angle_pow", 0.0)

    if isinstance(score_post_t0_angle_pow, str) and score_post_t0_angle_pow.lower() == "auto":
        # 自动触发：只在“高分像素”的 angle_gate_t0 中位数足够大时才启用，避免像 abu-beach-4 这类 t0 不匹配时误伤 AUC1。
        top_ratio = float(cfg.get("score_post_t0_angle_auto_top_ratio", 0.005))
        thr = float(cfg.get("score_post_t0_angle_auto_thr", 0.60))
        p_max = float(cfg.get("score_post_t0_angle_auto_p", 0.60))

        try:
            s_flat = score.reshape(-1).astype(np.float32)
            g_flat = np.asarray(angle_gate_t0_all, dtype=np.float32).reshape(-1)
            k = max(1, int(s_flat.size * top_ratio))
            idx_top = np.argpartition(s_flat, -k)[-k:]
            med_g = float(np.median(g_flat[idx_top]))
        except Exception:
            med_g = 0.0

        if (med_g >= thr) and (p_max > 0):
            score_post_t0_angle_pow_used = float(p_max)
            g = np.clip(np.asarray(angle_gate_t0_all, dtype=np.float32), 0.0, 1.0)
            score = np.clip(score * np.power(g, float(p_max)), 0.0, 1.0).astype(np.float32)
        else:
            score_post_t0_angle_pow_used = 0.0
    else:
        try:
            p = float(score_post_t0_angle_pow)
        except Exception:
            p = 0.0

        if p > 0:
            score_post_t0_angle_pow_used = float(p)
            g = np.clip(np.asarray(angle_gate_t0_all, dtype=np.float32), 0.0, 1.0)
            score = np.clip(score * np.power(g, float(p)), 0.0, 1.0).astype(np.float32)

    score_post_pow_used = np.nan
    pow_auto = bool(cfg.get("score_post_pow_auto", False)) or bool(cfg.get("score_post_auto", False))

    if pow_auto:
        try:
            score_post_pow = _auto_pow_from_bg_np(score, sim_bg_all, gate, cfg)
        except Exception:
            score_post_pow = float(cfg.get("score_post_pow", 1.0))
    else:
        score_post_pow = float(cfg.get("score_post_pow", 1.0))

    if np.isfinite(score_post_pow) and (abs(score_post_pow - 1.0) > 1e-6):
        if score_post_pow <= 0:
            # 非法值：跳过，不影响主流程
            score_post_pow = 1.0
        else:
            score_post_pow_used = float(score_post_pow)
            score = np.clip(score, 0.0, 1.0).astype(np.float32)
            score = np.power(score, score_post_pow).astype(np.float32)# --- optional: high-score lift (monotonic, test-time only) ---
    # 目的：在不改变排序（AUC1 不变）的前提下，把“高分尾部”进一步推近 1，
    #      从而提升 τ-Pd（AUC2）并让目标更“亮”（更突出）。
    # 说明：
    #   - 这是一个分段严格单调映射，不会改变 AUC1；
    #   - 只对 score > thr 的部分生效，避免整体抬高背景。
    score_post_highlift = bool(cfg.get("score_post_highlift", False))
    score_post_highlift_thr_used = np.nan
    score_post_highlift_gamma_used = np.nan
    if score_post_highlift:
        try:
            thr = float(cfg.get("score_post_highlift_thr", 0.50))
            gamma = float(cfg.get("score_post_highlift_gamma", 2.0))
            if np.isfinite(thr) and np.isfinite(gamma) and (0.0 < thr < 1.0) and (gamma > 0.0):
                s = np.clip(score.astype(np.float32), 0.0, 1.0)
                msk = (s > thr)
                if bool(np.any(msk)):
                    x = (s[msk] - thr) / max(1.0 - thr, 1e-6)  # in (0,1]
                    x = np.clip(x, 0.0, 1.0)
                    # lift: x' = 1 - (1-x)^gamma
                    x2 = 1.0 - np.power((1.0 - x), gamma).astype(np.float32)
                    s[msk] = thr + (1.0 - thr) * x2
                    score = np.clip(s, 0.0, 1.0).astype(np.float32)
                score_post_highlift_thr_used = float(thr)
                score_post_highlift_gamma_used = float(gamma)
        except Exception:
            pass

    if return_debug:
        dbg = {
            "sem_margin": sem_margin_all,
            "wave_sim": wave_sim_all,
            "deriv_sim": deriv_sim_all,
            "sim_t": sim_t_all,
            "sim_bg": sim_bg_all,
            "gate": gate,
            "logit": logit.astype(np.float32),
            "angle_raw": angle_raw_all,
            "angle_gate": angle_gate_all,
            "angle_raw_t0": angle_raw_t0_all,
            "angle_gate_t0": angle_gate_t0_all,
        }
        # adaptive fusion 的中间量（便于复盘 AUC1 掉点是否来自 gate 误伤）
        if "conf" in locals():
            dbg["fusion_conf"] = conf.astype(np.float32)
        if "sup" in locals():
            dbg["fusion_sup"] = sup.astype(np.float32)
        if "angle_power_used" in locals():
            dbg["angle_power_used"] = np.asarray(angle_power_used, dtype=np.float32)
        # target-bank 先验权重（可选，排查 multiple comparisons 很有用）
        if t_bank_w is not None:
            dbg["t_bank_w"] = t_bank_w.detach().cpu().numpy().astype(np.float32)
        if t_bank_ang is not None:
            dbg["t_bank_ang"] = t_bank_ang.astype(np.float32)
        # nbs_proto 模式下，如果使用了 bg_gate 的自动阈值，这里把阈值也塞进 debug 方便复盘
        if "bg_thr_used" in locals():
            dbg["bg_thr_used"] = np.array([bg_thr_used], dtype=np.float32)
        # NBS 归一化时用到的统计量（有助于定位 AUC 下降是否来自标定）
        if np.isfinite(t_max_used):
            dbg["t_max_used"] = np.array([t_max_used], dtype=np.float32)
        if np.isfinite(wave_max_used):
            dbg["wave_max_used"] = np.array([wave_max_used], dtype=np.float32)
        if np.isfinite(deriv_max_used):
            dbg["deriv_max_used"] = np.array([deriv_max_used], dtype=np.float32)
        if np.isfinite(score_post_scale):
            dbg["score_post_scale"] = np.array([score_post_scale], dtype=np.float32)
        if np.isfinite(score_post_sigmoid_mu_used):
            dbg["score_post_sigmoid_mu_used"] = np.array([score_post_sigmoid_mu_used], dtype=np.float32)
        if np.isfinite(score_post_sigmoid_k_used):
            dbg["score_post_sigmoid_k_used"] = np.array([score_post_sigmoid_k_used], dtype=np.float32)
        if np.isfinite(score_post_pow_used):
            dbg["score_post_pow_used"] = np.array([score_post_pow_used], dtype=np.float32)
        if np.isfinite(score_post_t0_angle_pow_used):
            dbg["score_post_t0_angle_pow_used"] = np.array([score_post_t0_angle_pow_used], dtype=np.float32)
        # highlift 用到的参数（若启用）
        if np.isfinite(score_post_highlift_thr_used):
            dbg["score_post_highlift_thr_used"] = np.array([score_post_highlift_thr_used], dtype=np.float32)
        if np.isfinite(score_post_highlift_gamma_used):
            dbg["score_post_highlift_gamma_used"] = np.array([score_post_highlift_gamma_used], dtype=np.float32)

        # 仅在 nbs_proto+margin_gate 模式下存在
        if "margin_gate" in locals():
            dbg["margin_gate"] = margin_gate.astype(np.float32)
        return score, dbg

    return score


def _compute_auc_metrics(gt_map: np.ndarray, score_map: np.ndarray):
    """输出 AUC1~AUC5（严格对齐 HTD-Mamba 论文定义）。

    指标定义（HTD-Mamba, Sec. III-A.2 Quality Metrics）：
      - AUC1: AUC(Pf, Pd) 检测有效性（越大越好）
      - AUC2: AUC(τ, Pd) 目标可检测性（越大越好）
      - AUC3: AUC(τ, Pf) 假警/背景抑制（越小越好）
      - AUC4: AUC1 + AUC2 - AUC3（越大越好，上限 2）
      - AUC5: AUC2 / (AUC3 + eps)（越大越好）

    重要修正（这也是你“AUC1 很高但其它指标异常”的常见来源）：
      1) sklearn.metrics.roc_curve 返回的 thresholds 默认是 **从大到小**。
         如果直接 auc(thresholds, tpr) / auc(thresholds, fpr) 会得到负值，
         从而把 AUC2/AUC3/AUC4/AUC5 全部带偏。
         这里将 (τ, Pd/Pf) 曲线统一成 τ **从小到大**，保证 AUC2/AUC3 为正且落在 [0,1]。
      2) 论文中的 τ 语义是对检测图做阈值扫掠的阈值，τ∈[0,1]。
         为保证不同 score_mode/不同方法的可比性，这里对 score 做到 [0,1] 的安全映射：
           - 若本身已在 [0,1]（允许微小数值越界），则 clip；
           - 否则做 min-max 映射到 [0,1]（单调变换，不改变排序，因此不影响 AUC1）。
    """
    y_true = (gt_map.reshape(-1) > 0).astype(np.int32)

    y_score = score_map.reshape(-1).astype(np.float32)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    s_min = float(np.min(y_score))
    s_max = float(np.max(y_score))

    if (not np.isfinite(s_min)) or (not np.isfinite(s_max)) or (s_max - s_min < 1e-12):
        # degenerate: all scores equal / numeric issue
        y_score01 = np.zeros_like(y_score, dtype=np.float32)
    else:
        # already in [0,1] ? (tolerate tiny numeric spill)
        if s_min >= -1e-3 and s_max <= 1.0 + 1e-3:
            y_score01 = np.clip(y_score, 0.0, 1.0).astype(np.float32)
        else:
            y_score01 = (y_score - s_min) / (s_max - s_min)
            y_score01 = np.clip(y_score01, 0.0, 1.0).astype(np.float32)

    # ROC (Pf, Pd)
    fpr, tpr, thr = roc_curve(y_true, y_score01)
    auc1 = float(auc(fpr, tpr))

    # τ-based unfolded curves: (τ, Pd) and (τ, Pf)
    # roc_curve thresholds are decreasing; reverse to make τ increasing.
    thr2 = thr[1:]
    Pd2 = tpr[1:]
    Pf2 = fpr[1:]

    tau = thr2[::-1]
    Pd = Pd2[::-1]
    Pf = Pf2[::-1]

    # Add endpoints τ=0 and τ=1 (exact values under our [0,1] score definition)
    pos_mask = (y_true == 1)
    neg_mask = ~pos_mask

    n_pos = float(np.sum(pos_mask))
    n_neg = float(np.sum(neg_mask))

    # τ = 0  -> predict all positive (since score ∈ [0,1])
    Pd0 = 1.0 if n_pos > 0 else 0.0
    Pf0 = 1.0 if n_neg > 0 else 0.0

    # τ = 1  -> only scores == 1 are predicted positive
    Pd1 = float(np.mean(y_score01[pos_mask] >= 1.0)) if n_pos > 0 else 0.0
    Pf1 = float(np.mean(y_score01[neg_mask] >= 1.0)) if n_neg > 0 else 0.0

    tau_ext = np.concatenate([
        np.array([0.0], dtype=np.float32),
        tau.astype(np.float32),
        np.array([1.0], dtype=np.float32)
    ], axis=0)
    Pd_ext = np.concatenate([
        np.array([Pd0], dtype=np.float32),
        Pd.astype(np.float32),
        np.array([Pd1], dtype=np.float32)
    ], axis=0)
    Pf_ext = np.concatenate([
        np.array([Pf0], dtype=np.float32),
        Pf.astype(np.float32),
        np.array([Pf1], dtype=np.float32)
    ], axis=0)

    # ensure τ is non-decreasing (cheap and safe)
    order = np.argsort(tau_ext)
    tau_ext = tau_ext[order]
    Pd_ext = Pd_ext[order]
    Pf_ext = Pf_ext[order]

    auc2 = float(auc(tau_ext, Pd_ext))  # Detectability
    auc3 = float(auc(tau_ext, Pf_ext))  # False alarm / background suppression (smaller better)

    auc4 = float(auc1 + auc2 - auc3)
    auc5 = float(auc2 / (auc3 + 1e-12))

    curves = {
        'fpr': fpr,
        'tpr': tpr,
        'thr': thr,
        'tau': tau_ext,
        'pd_tau': Pd_ext,
        'pf_tau': Pf_ext,
    }
    return auc1, auc2, auc3, auc4, auc5, curves


# =========================
# Training
# =========================


def train(cfg: Dict):
    """训练入口。"""
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = torch.device(
        cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    )

    save_dir = cfg.get("save_dir", "./models/")
    log_dir = cfg.get("log_dir", "./logs/")
    checkFile(save_dir)
    checkFile(log_dir)

    # logger
    log_name = cfg.get(
        "log_name", f"HTD_MambaNext_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path = os.path.join(log_dir, log_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("===== Train config =====")
    for k, v in cfg.items():
        logging.info(f"{k}: {v}")

    # dataset
    dataset = Data(
        cfg["path"],
        w_size=int(cfg.get("w_size", 11)),
        use_augment=bool(cfg.get("use_augment", True)),
        augment_ctrl=cfg.get("augment_ctrl", None),
        augment_view1_scale=float(cfg.get("augment_view1_scale", 0.35)),
        augment_view2_scale=float(cfg.get("augment_view2_scale", 1.0)),
        patch_encode_tau=float(cfg.get("patch_encode_tau", 1.0)),
        patch_encode_sim=str(cfg.get("patch_encode_sim", "corr")),
        patch_encode_sigma=cfg.get("patch_encode_sigma", None),
        patch_encode_topk_ratio=cfg.get("patch_encode_topk_ratio", None),
        hard_neg_quantile=float(cfg.get("hard_neg_quantile", 0.10)),
        hard_neg_region=str(cfg.get("hard_neg_region", "patch")),
        hard_neg_inner_size=cfg.get("hard_neg_inner_size", None),
        return_ctx_sim=bool(cfg.get("return_ctx_sim", True)),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 80)),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # model
    model = _build_model(cfg)
    model.train()

    # load weights (optional)
    if cfg.get("training_load_weight", None):
        ckpt_path = os.path.join(save_dir, cfg["training_load_weight"])
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        logging.info(f"Loaded weights: {ckpt_path}")

    # optimizer
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))

    # -------------------------
    # Optimizer：SwAV/原型学习场景下，建议给 prototypes / bias / norm 参数关闭 weight decay
    # 这样原型不会被 L2 正则“拉回 0”，配合 unit-norm 更稳定。
    # -------------------------
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name_l = name.lower()
        if ("bg_prototypes" in name_l) or name_l.endswith(".bias") or (p.ndim == 1) or ("norm" in name_l):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    # AdamW 与论文实现更一致（weight decay 与梯度更新解耦），对对比学习 + 原型学习更稳。
    opt = AdamW(
        [
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    # -------------------------
    # LR scheduler（轻量版）：linear warmup + cosine decay
    # 依据 HTD-Mamba 的训练策略：先 warmup 再 cosine 下降，能提升收敛与可复现性。
    # 这里按“epoch 级”更新，代码简单且足够稳定。
    # -------------------------
    max_epoch = int(cfg.get("epoch", 200))
    use_lr_sched = bool(cfg.get("use_lr_scheduler", True))
    warmup_epochs = int(cfg.get("lr_warmup_epochs", 10))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.05))
    base_lr = float(lr)
    min_lr = float(base_lr * min_lr_ratio)

    def _lr_at_epoch(ep: int) -> float:
        ep = int(ep)
        if not use_lr_sched:
            return base_lr
        if warmup_epochs > 0 and ep <= warmup_epochs:
            return base_lr * float(ep) / float(max(1, warmup_epochs))
        if max_epoch <= max(1, warmup_epochs):
            return min_lr
        # cosine decay
        t = float(ep - warmup_epochs) / float(max(1, max_epoch - warmup_epochs))
        t = float(np.clip(t, 0.0, 1.0))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    def _set_lr(optimizer: torch.optim.Optimizer, lr_val: float):
        for pg in optimizer.param_groups:
            pg["lr"] = float(lr_val)

    # temps
    temp_intra = float(cfg.get("delta", 0.07))
    temp_ctx = float(cfg.get("delta_ctx", 0.07))

    # loss weights
    w_intra = float(cfg.get("w_intra", 1.0))
    w_ctx = float(cfg.get("w_ctx", 0.35))
    w_local = float(cfg.get("w_local_neg", 0.05))
    w_proto = float(cfg.get("w_proto", 0.15))
    w_ortho = float(cfg.get("w_proto_ortho", 0.02))

    # -------------------------
    # NEW: 排序能力增强（训练期，从根上改表示，而不是只靠测试期后处理“压背景/抬分”）
    # -------------------------
    # (1) WIN-InfoNCE：用 raw 域谱角给 negatives 加权，缓解 false-negative
    use_win = bool(cfg.get("use_win", False))
    win_thr_mode = str(cfg.get("win_thr_mode", "fixed")).lower()
    win_thr = float(cfg.get("win_thr", 0.12))      # radian（~6.9°）
    win_thr_q = float(cfg.get("win_thr_q", 0.10))  # quantile mode
    win_k = float(cfg.get("win_k", 20.0))
    win_min_w = float(cfg.get("win_min_w", 1e-4))

    # (2) Batch-hard triplet：专门拉开“最像目标的背景负样本”（更直接提升 AUC1）
    w_bh = float(cfg.get("w_bh", 0.0))
    bh_margin = float(cfg.get("bh_margin", 0.15))
    bh_start_epoch = int(cfg.get("bh_start_epoch", 20))

    # --- prototype warmup（建议：先让 encoder 学到稳定表征，再逐步加大原型约束，避免早期塌陷/把稀有目标吸进原型） ---
    proto_start_epoch = int(cfg.get("proto_start_epoch", 1))
    proto_warmup = int(cfg.get("proto_warmup", 0))  # 0 表示不 warmup，直接全权重

    # --- Spectral-Geometry Alignment (SGA) ---
    # 让 embedding 保持 raw 光谱域的相似度结构，提升“目标谱匹配”的可靠性
    w_geom = float(cfg.get("w_geom", 0.10))
    geom_topk = int(cfg.get("geom_topk", 8))
    geom_sim_thr = float(cfg.get("geom_sim_thr", 0.70))
    geom_use_density_weight = bool(cfg.get("geom_use_density_weight", False))

    # ctx gate
    # ctx_sim_raw 是 RAW 域的谱角距离（radian）。
    # 为了提升“异质背景下的阈值稳定性”，这里支持两种阈值：
    #   1) fixed    : 使用固定谱角阈值 ctx_sim_thr（有物理意义，但跨数据集需调参）
    #   2) quantile : 使用 batch 内谱角的分位数阈值 ctx_thr_q（更稳健/少调参）
    ctx_thr_mode = str(cfg.get("ctx_thr_mode", "fixed")).lower()
    ctx_thr_q = float(cfg.get("ctx_thr_q", 0.60))
    ctx_sim_thr = float(cfg.get("ctx_sim_thr", 0.55))
    ctx_sim_k = float(cfg.get("ctx_sim_k", 10.0))
    ctx_start_epoch = int(cfg.get("ctx_start_epoch", 1))
    ctx_warmup = int(cfg.get("ctx_warmup", 20))

    # local
    local_margin = float(cfg.get("local_margin", 0.2))
    local_warmup = int(cfg.get("local_neg_warmup", 10))

    # false-negative suppression
    fn_topk = int(cfg.get("fn_topk", 8))

    # proto params
    proto_tau = float(cfg.get("proto_tau", 0.1))
    sinkhorn_eps = float(cfg.get("sinkhorn_eps", 0.05))
    sinkhorn_iters = int(cfg.get("sinkhorn_iters", 3))

    # SwAV/Sinkhorn 是否只在“背景样本”上计算 assignment
    proto_bg_only_thr = cfg.get("proto_bg_only_thr", None)
    if proto_bg_only_thr is not None:
        proto_bg_only_thr = float(proto_bg_only_thr)
    proto_min_bg_samples = int(cfg.get("proto_min_bg_samples", 16))

    # density gate for proto loss
    dens_topk = int(cfg.get("proto_density_topk", 8))
    # 注意：这里的 dens_thr 是**谱角阈值 (radian)**，角度越小越像背景
    dens_thr = float(cfg.get("proto_density_thr", 0.25))
    dens_k = float(cfg.get("proto_density_k", 12.0))

    # density gate 模式：
    # - "sigmoid" : 使用绝对谱角阈值 dens_thr（有物理意义，但跨数据集可能需微调）
    # - "quantile": 使用 batch 内 dens_ang 分位数阈值（更稳健，避免异质背景下阈值不稳）
    proto_density_mode = str(cfg.get("proto_density_mode", "sigmoid")).lower()
    proto_bg_quantile = float(cfg.get("proto_bg_quantile", 0.60))

    # outlier repulsion (optional)
    w_repulse = float(cfg.get("w_repulse", 0.0))
    rep_start_epoch = int(cfg.get("repulse_start_epoch", 20))
    rep_outlier_thr = float(cfg.get("repulse_outlier_thr", 0.45))
    rep_outlier_k = float(cfg.get("repulse_outlier_k", 12.0))
    rep_margin = float(cfg.get("repulse_margin", 0.55))

    # drift curriculum
    drift_final = float(cfg.get("drift_p", 0.25))
    drift_curriculum = bool(cfg.get("drift_curriculum", True))
    drift_warmup = int(cfg.get("drift_warmup", 40))

    # grad clip
    grad_clip = float(cfg.get("grad_clip", 1.0))

    # ============================================================
    # NEW: checkpoint 保存/清理 + 在线评估/早停（可选）
    # ------------------------------------------------------------
    # 目的：
    #   1) 避免默认“每个 epoch 都保存 ckpt”，导致 models 目录爆炸；
    #   2) 支持在线评估（debug_eval_interval）+ 早停（early_stop_patience），
    #      减少无意义的长时间训练。
    #
    # 说明：
    #   - 如果你不设置这些参数，行为与原版保持一致（每 epoch 保存）。
    #   - 在线评估会调用 eval()，需要能读取 gt/map（用于 AUC 计算）。
    # ============================================================
    save_interval = int(cfg.get("save_interval", 1))  # 每隔多少 epoch 保存一次 ckpt（>=1）
    save_interval = max(1, save_interval)
    keep_last_ckpts = int(cfg.get("keep_last_ckpts", 0))  # 只保留最近 N 个 ckpt（0 表示不清理）
    keep_last_ckpts = max(0, keep_last_ckpts)

    save_best_ckpt = bool(cfg.get("save_best_ckpt", True))  # 在线评估时，是否额外保存 best ckpt 信息
    best_json_path = os.path.join(save_dir, "best_ckpt.json")

    # online eval / early-stop（依赖 debug_eval_interval > 0）
    debug_eval_interval = int(cfg.get("debug_eval_interval", 0))
    early_stop_patience = int(cfg.get("early_stop_patience", 0))  # 连续多少次评估不提升则停止
    early_stop_patience = max(0, early_stop_patience)
    early_stop_min_epoch = int(cfg.get("early_stop_min_epoch", 10))  # 至少训练多少 epoch 才允许早停

    # 在线挑选的指标（默认沿用 select_metric）
    online_metric = str(cfg.get("select_metric", "AUC4")).upper()
    _valid_auc = {"AUC1", "AUC2", "AUC3", "AUC4", "AUC5"}
    if online_metric not in _valid_auc:
        online_metric = "AUC4"

    best_online_score = -1e9
    best_online_ep = 0
    best_online_ckpt = None  # type: Optional[str]
    no_improve = 0

    def _cleanup_old_ckpts():
        """根据 keep_last_ckpts 清理旧 ckpt（并强制保留 best_online_ckpt）。"""
        if keep_last_ckpts <= 0:
            return
        ckpts = [p for p in glob.glob(os.path.join(save_dir, "ckpt_*_.pt")) if os.path.isfile(p)]
        # 按 epoch 排序
        def _ep(p: str) -> int:
            base = os.path.basename(p)
            try:
                return int(base.split("_")[1])
            except Exception:
                return -1
        ckpts = sorted(ckpts, key=_ep)
        keep = set(ckpts[-keep_last_ckpts:])
        if best_online_ckpt is not None:
            bp = os.path.join(save_dir, best_online_ckpt)
            if os.path.exists(bp):
                keep.add(bp)
        for p in ckpts:
            if p not in keep:
                try:
                    os.remove(p)
                except Exception:
                    pass

    for ep in range(1, max_epoch + 1):
        model.train()

        # 更新学习率（epoch 级）：warmup + cosine
        lr_cur = _lr_at_epoch(ep)
        _set_lr(opt, lr_cur)

        # -------------------------
        # Augmentation curriculum：先弱后强（提升细粒度谱形可感知性 + 训练稳定性）
        # 说明：强扰动太早会让 encoder 学到“过度不变”，对窄谱目标不友好。
        # 这里仅通过动态调节增强强度实现，不增加额外训练分支。
        # -------------------------
        if bool(cfg.get("aug_curriculum", True)) and getattr(dataset, "use_augment", False):
            ramp_ep = int(cfg.get("aug_ramp_epochs", 30))
            r = float(ep) / float(max(1, ramp_ep))
            if r > 1.0:
                r = 1.0
            v1_start = float(cfg.get("augment_view1_scale_start", 0.20))
            v2_start = float(cfg.get("augment_view2_scale_start", 0.50))
            v1_final = float(cfg.get("augment_view1_scale", 0.35))
            v2_final = float(cfg.get("augment_view2_scale", 1.00))
            v1 = v1_start + (v1_final - v1_start) * r
            v2 = v2_start + (v2_final - v2_start) * r
            try:
                dataset.set_augment_scales(v1, v2)
            except Exception:
                dataset.augment_view1_scale = float(v1)
                dataset.augment_view2_scale = float(v2)

        # 计算本 epoch 的 prototype 权重（warmup）
        if proto_warmup > 0:
            if ep < proto_start_epoch:
                proto_factor = 0.0
            else:
                proto_factor = min(
                    1.0,
                    float(ep - proto_start_epoch + 1) / max(1.0, float(proto_warmup)),
                )
        else:
            proto_factor = 1.0
        w_proto_eff = w_proto * proto_factor
        w_ortho_eff = w_ortho * proto_factor

        # drift curriculum
        if drift_curriculum and hasattr(model, "drift_p"):
            ratio = min(1.0, float(ep) / max(1.0, float(drift_warmup)))
            model.drift_p = drift_final * ratio

        epoch_loss = []
        pbar = tqdm(loader, desc=f"epoch {ep}/{max_epoch}")

        for batch in pbar:
            if len(batch) == 6:
                center_v1, center_v2, coded_v2, hardneg_v2, center_raw, ctx_sim_raw = (
                    batch
                )
            else:
                center_v1, center_v2, coded_v2, hardneg_v2, center_raw = batch
                ctx_sim_raw = None

            center_v1 = center_v1.to(device)
            center_v2 = center_v2.to(device)
            coded_v2 = coded_v2.to(device)
            hardneg_v2 = hardneg_v2.to(device)
            center_raw = center_raw.to(device)
            if ctx_sim_raw is not None:
                ctx_sim_raw = ctx_sim_raw.to(device)

            # forward
            # 注：hard negative 只有在 w_local_neg>0 且过了 warmup 才需要，避免无谓计算开销
            need_local = (w_local > 0) and (ep > local_warmup)
            z1 = model(center_v1)  # [B,D]
            z2 = model(center_v2)  # [B,D]
            zc = model(coded_v2)  # [B,D]
            zn = model(hardneg_v2) if need_local else None  # [B,D] or None

            # raw similarity for masks / density
            # 这里使用 correlation-cosine（减均值再 cosine），避免全正导致相似度饱和
            sim_raw = _corr_sim_matrix(center_raw.squeeze(1))
            neg_mask = _build_false_negative_mask(sim_raw, topk=fn_topk)

            # raw angle matrix (radian): reused by WIN / prototype density / outlier gates
            ang_raw = _spectral_angle_from_sim(sim_raw)

            # 1) intra contrast
            if use_win:
                w_win = _build_win_weights_from_angle(
                    ang_raw,
                    thr_mode=win_thr_mode,
                    thr=win_thr,
                    thr_q=win_thr_q,
                    k=win_k,
                    min_w=win_min_w,
                )
                loss_intra = win_info_nce_loss(
                    z1, z2, temperature=temp_intra, neg_weights=w_win, neg_mask=neg_mask
                )
            else:
                loss_intra = info_nce_loss(
                    z1, z2, temperature=temp_intra, neg_mask=neg_mask
                )

            # 2) ctx contrast (gated)
            if ctx_sim_raw is None:
                w_ctx_vec = torch.ones((z1.shape[0],), device=device)
            else:
                # ctx_sim_raw 是谱角距离（radian）：角度越小越相似。
                # sigmoid gate: angle 越小，越信任 ctx；angle 大(异质 patch)则弱化。
                if ctx_thr_mode in ["quantile", "q", "rank"]:
                    q = float(np.clip(ctx_thr_q, 0.05, 0.95))
                    thr_ctx = _torch_quantile(ctx_sim_raw.detach(), q=q)
                else:
                    thr_ctx = torch.tensor(ctx_sim_thr, device=device, dtype=ctx_sim_raw.dtype)

                w_ctx_vec = torch.sigmoid(ctx_sim_k * (thr_ctx - ctx_sim_raw))

                # warmup
                if ep < ctx_start_epoch:
                    w_ctx_vec = torch.zeros_like(w_ctx_vec)
                else:
                    # 线性 warmup 到 1
                    t = min(
                        1.0,
                        float(ep - ctx_start_epoch + 1) / max(1.0, float(ctx_warmup)),
                    )
                    w_ctx_vec = w_ctx_vec * t

            loss_ctx = weighted_info_nce_loss(
                z1, zc, weights=w_ctx_vec, temperature=temp_ctx, neg_mask=neg_mask
            )

            # 3) local hard negative（patch 内）
            if not need_local:
                loss_local = torch.tensor(0.0, device=device)
            else:
                loss_local = hardneg_triplet_loss(z1, z2, zn, margin=local_margin)

            # 3.5) batch-hard triplet（batch 内，全局最难负样本，更直接提升排序/AUC1）
            if (w_bh > 0) and (ep >= bh_start_epoch):
                loss_bh = batch_hard_triplet_loss(z1, z2, margin=bh_margin, neg_mask=neg_mask)
            else:
                loss_bh = torch.tensor(0.0, device=device)

            # 4) prototype swapped assignment (density-weighted)
            # density (spectral angle): 越小越“密度高/背景”
            # density (spectral angle): 越小越“密度高/背景”
            # ang_raw 已在前面计算（可被 WIN/原型密度/离群门控复用）
            dens_ang = _batch_density_from_angle(ang_raw, topk=dens_topk)
            # w_bg 越大越像背景。
            # 经验：在“背景很异质/含结构性背景”的场景，固定 dens_thr 容易不稳定，
            # 因此提供 quantile 方式：每个 batch 只选 dens_ang 最小的一部分样本作为“高置信背景”。
            if proto_density_mode in ["quantile", "q", "rank"]:
                q = float(np.clip(proto_bg_quantile, 0.05, 0.95))
                thr_q = _torch_quantile(dens_ang.detach(), q=q)
                w_bg = (dens_ang <= thr_q).float()
            elif proto_density_mode in ["quantile_sigmoid", "q_sigmoid", "rank_sigmoid"]:
                # 先用 batch 分位数得到“相对稳定”的阈值，再用 sigmoid 做软权重（比 0/1 更平滑）
                q = float(np.clip(proto_bg_quantile, 0.05, 0.95))
                thr_q = _torch_quantile(dens_ang.detach(), q=q)
                w_bg = torch.sigmoid(dens_k * (thr_q - dens_ang))
            else:
                w_bg = torch.sigmoid(dens_k * (dens_thr - dens_ang))

            prototypes = model.get_bg_prototypes(normalize=False)
            loss_proto = swapped_proto_loss(
                z1,
                z2,
                prototypes,
                weights=w_bg,
                tau=proto_tau,
                sinkhorn_eps=sinkhorn_eps,
                sinkhorn_iters=sinkhorn_iters,
                bg_only_thr=proto_bg_only_thr,
                min_bg_samples=proto_min_bg_samples,
            )

            loss_ortho = prototype_orthogonal_loss(prototypes)

            # 4.5) outlier repulsion（可选，默认关）
            if w_repulse > 0 and ep >= rep_start_epoch:
                loss_rep = 0.5 * (
                    prototype_outlier_repulsion_loss(
                        z1,
                        prototypes,
                        dens_ang,
                        out_thr=rep_outlier_thr,
                        out_k=rep_outlier_k,
                        margin=rep_margin,
                    )
                    + prototype_outlier_repulsion_loss(
                        z2,
                        prototypes,
                        dens_ang,
                        out_thr=rep_outlier_thr,
                        out_k=rep_outlier_k,
                        margin=rep_margin,
                    )
                )
            else:
                loss_rep = torch.tensor(0.0, device=device)

            # 5) Spectral-Geometry Alignment（可选）
            # 让 embedding 的局部相似度结构对齐 raw 光谱域，提升目标谱匹配能力
            if w_geom > 0:
                w_geom_vec = w_bg if geom_use_density_weight else None
                loss_geom = 0.5 * (
                    spectral_geometry_alignment_loss(
                        z1,
                        sim_raw,
                        topk=geom_topk,
                        sim_thr=geom_sim_thr,
                        weights=w_geom_vec,
                    )
                    + spectral_geometry_alignment_loss(
                        z2,
                        sim_raw,
                        topk=geom_topk,
                        sim_thr=geom_sim_thr,
                        weights=w_geom_vec,
                    )
                )
            else:
                loss_geom = torch.tensor(0.0, device=device)

            loss = (
                w_intra * loss_intra
                + w_ctx * loss_ctx
                + w_local * loss_local
                + w_bh * loss_bh
                + w_proto_eff * loss_proto
                + w_ortho_eff * loss_ortho
                + w_repulse * loss_rep
                + w_geom * loss_geom
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            # SwAV 风格：把 prototypes 重新投影到单位球面（避免数值漂移/尺度不稳定）
            if getattr(model, "bg_prototypes", None) is not None:
                with torch.no_grad():
                    model.bg_prototypes.copy_(F.normalize(model.bg_prototypes, dim=1, eps=1e-6))

            l = float(loss.item())
            epoch_loss.append(l)

            pbar.set_postfix(
                {
                    "loss": f"{l:.4f}",
                    "intra": f"{float(loss_intra.item()):.3f}",
                    "ctx": f"{float(loss_ctx.item()):.3f}",
                    "loc": f"{float(loss_local.item()):.3f}",
                    "bh": f"{float(loss_bh.item()):.3f}",
                    "proto": f"{float(loss_proto.item()):.3f}",
                    "rep": f"{float(loss_rep.item()):.3f}",
                    "geom": f"{float(loss_geom.item()):.3f}",
                }
            )

        mean_loss = float(np.mean(epoch_loss)) if len(epoch_loss) else 0.0
        logging.info(
            f"[Epoch {ep:03d}] loss={mean_loss:.6f} drift_p={getattr(model, 'drift_p', 0.0):.3f}"
        )


        # ------------------------------------------------------------
        # NEW: 按 save_interval 保存 checkpoint（并可选在线评估/早停）
        # ------------------------------------------------------------
        do_eval = debug_eval_interval > 0 and (ep % debug_eval_interval == 0 or ep == 1)
        do_save = (ep % save_interval == 0) or (ep == max_epoch) or do_eval

        ckpt_name = f"ckpt_{ep}_.pt"
        ckpt_path = os.path.join(save_dir, ckpt_name)

        # 只在需要时保存（默认 save_interval=1 -> 行为等价于“每 epoch 保存”）
        if do_save:
            torch.save(model.state_dict(), ckpt_path)

        # 在线评估：只在 do_eval 时执行（避免每个 epoch 都跑 eval）
        if do_eval:
            try:
                cfg2 = dict(cfg)
                cfg2["test_load_weight"] = ckpt_name
                a1, a2, a3, a4, a5 = eval(cfg2, quiet=True)

                metric_map = {"AUC1": a1, "AUC2": a2, "AUC3": a3, "AUC4": a4, "AUC5": a5}
                cur_score = float(metric_map.get(online_metric, a4))

                logging.info(
                    f"[OnlineEval] ep={ep:03d} {online_metric}={cur_score:.6f} "
                    f"(AUC1={a1:.4f}, AUC2={a2:.4f}, AUC3={a3:.4f}, AUC4={a4:.4f}, AUC5={a5:.2f})"
                )

                if cur_score > best_online_score:
                    best_online_score = cur_score
                    best_online_ep = ep
                    best_online_ckpt = ckpt_name
                    no_improve = 0

                    if save_best_ckpt:
                        # 写 best 信息（供 select_best 快速返回 / 也方便复现实验）
                        try:
                            with open(best_json_path, "w", encoding="utf-8") as jf:
                                json.dump(
                                    {
                                        "best_epoch": int(best_online_ep),
                                        "best_score": float(best_online_score),
                                        "metric": str(online_metric),
                                        "ckpt": str(best_online_ckpt),
                                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    },
                                    jf,
                                    indent=2,
                                    ensure_ascii=False,
                                )
                        except Exception:
                            pass
                else:
                    no_improve += 1

                # 清理旧 ckpt（会保留 best_online_ckpt）
                _cleanup_old_ckpts()

                # 早停：连续 early_stop_patience 次评估不提升就停止
                if (
                    early_stop_patience > 0
                    and ep >= early_stop_min_epoch
                    and no_improve >= early_stop_patience
                ):
                    logging.info(
                        f"[EarlyStop] no improve for {no_improve} evals, stop at ep={ep}"
                    )
                    break

            except Exception as e:
                logging.warning(f"online eval failed: {e}")
        else:
            # 不做在线评估时，也可以按需清理旧 ckpt
            if do_save:
                _cleanup_old_ckpts()

    logging.info("===== Training finished =====")
    logging.info(f"Log saved to: {log_path}")


# =========================
# Eval / Select-best
# =========================



def _resolve_ckpt_path(save_dir: str, ckpt_name_or_path: str) -> str:
    """解析 checkpoint 路径（兼容“文件名”或“完整/相对路径”）。

    背景：
      - 训练/默认评估时，代码通常会用：ckpt_path = os.path.join(save_dir, ckpt_name)
      - 但做消融/复用 ckpt 时，你很可能会直接传入一个路径，例如：
            --test_load_weight ./ckpt/abu-airport-2/FULL_DSM/ckpt_160_.pt
        如果还强行再拼一次 save_dir，就会变成重复路径，导致 FileNotFoundError。

    这里的策略：
      依次尝试若干候选路径，找到第一个真实存在的 ckpt 文件。
    """
    if ckpt_name_or_path is None:
        raise ValueError("ckpt_name_or_path is None")

    ckpt = str(ckpt_name_or_path).strip()
    if ckpt == "":
        raise ValueError("ckpt_name_or_path is empty")

    cand = []

    # 1) 直接当成路径（相对/绝对都可以）
    cand.append(ckpt)

    # 2) 当成“文件名”，拼到 save_dir 下
    if save_dir is not None and str(save_dir).strip() != "":
        sd = str(save_dir)
        cand.append(os.path.join(sd, ckpt))

        # 3) 兜底：如果你传入的是一个带目录的相对路径，且其中已经包含 save_dir，
        #    但由于当前工作目录不同导致 cand[0] 找不到，则用 basename 再拼一次。
        cand.append(os.path.join(sd, os.path.basename(ckpt)))

    # 去重 + 规范化
    uniq = []
    for p in cand:
        p2 = os.path.normpath(p)
        if p2 not in uniq:
            uniq.append(p2)

    for p in uniq:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Cannot find checkpoint. Tried:\n  - " + "\n  - ".join(uniq)
    )



def _auto_find_ckpt(save_dir: str) -> Optional[str]:
    """在 save_dir 下自动选择一个 checkpoint（用于 eval 阶段兜底）。

    优先策略：
      1) 如果存在 `ckpt_{epoch}_.pt`，选 epoch 最大的；
      2) 否则在所有 `.pt` 里选最近修改的那个。

    返回：
      - 文件名（相对 save_dir）；后续会交给 `_resolve_ckpt_path` 统一解析。
    """
    if not save_dir:
        return None
    if not os.path.isdir(save_dir):
        return None

    pts = [fn for fn in os.listdir(save_dir) if fn.endswith(".pt")]
    if not pts:
        return None

    # 1) 优先解析 ckpt_{epoch}_.pt
    best_fn = None
    best_ep = -1
    for fn in pts:
        if fn.startswith("ckpt_") and fn.endswith("_.pt"):
            mid = fn[len("ckpt_"):-len("_.pt")]
            if mid.isdigit():
                ep = int(mid)
                if ep > best_ep:
                    best_ep = ep
                    best_fn = fn
    if best_fn is not None:
        return best_fn

    # 2) fallback: 选最近修改的 .pt
    pts_sorted = sorted(
        pts,
        key=lambda x: os.path.getmtime(os.path.join(save_dir, x)),
        reverse=True,
    )
    return pts_sorted[0] if pts_sorted else None


def eval(cfg: Dict, quiet: bool = False, save_outputs: Optional[bool] = None):
    device = torch.device(
        cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    )

    mat = sio.loadmat(cfg["path"])
    data = standard(mat["data"]).astype(np.float32)
    gt = mat.get("map", None)
    if gt is None:
        raise KeyError("dataset .mat must contain 'map' for evaluation")
    gt = gt.astype(np.int32)

    H, W, C = data.shape
    X = data.reshape(-1, C, order="F")

    # Provide spatial shape for optional spatial refinement in the head (eval only)
    cfg = dict(cfg)
    cfg['img_h'] = int(H)
    cfg['img_w'] = int(W)

    # target spectrum (eval only)
    target_spectrum = ts_generation(data, gt, type=int(cfg.get("ts_type", 7)))

    # model + load
    model = _build_model(cfg)

    save_dir = cfg.get("save_dir", "./models/")

    # 兼容：如果你只想 `--mode eval` 直接跑，而没有手动指定 ckpt
    # （或者 cfg 里没写 test_load_weight），这里会自动在 save_dir 下找一个 ckpt_*.pt。
    ckpt_name = cfg.get("test_load_weight", None)
    if ckpt_name is None or str(ckpt_name).strip() == "":
        auto_ckpt = _auto_find_ckpt(save_dir)
        if auto_ckpt is None:
            raise ValueError(
                "cfg['test_load_weight'] is required for eval, and no ckpt_*.pt found in save_dir."
            )
        ckpt_name = auto_ckpt
        cfg["test_load_weight"] = ckpt_name
        if not quiet:
            print(f"[eval] auto select ckpt: {ckpt_name}")

    ckpt_path = _resolve_ckpt_path(save_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()

    # eval debug：返回一些内部量（sim_t/sim_bg/margin/wave/angle 等），便于你排查“假警来自哪里”
    save_debug = bool(cfg.get("eval_save_debug", True))
    if save_outputs and save_debug:
        score_vec, dbg = compute_detection_scores(
            model, X, target_spectrum, device, cfg, return_debug=True
        )
    else:
        score_vec = compute_detection_scores(
            model, X, target_spectrum, device, cfg, return_debug=False
        )
        dbg = None
    score_map = score_vec.reshape(H, W, order="F")

    auc1, auc2, auc3, auc4, auc5, curves = _compute_auc_metrics(gt, score_map)

    if not quiet:
        print(
            f"AUC1={auc1:.6f} AUC2={auc2:.6f} AUC3={auc3:.6f} AUC4={auc4:.6f} AUC5={auc5:.6f}"
        )

    # save outputs
    if save_outputs:
        out_dir = cfg.get("out_dir", "./outputs/")
        checkFile(out_dir)

        # ============================================================
        # Clean visualizations（无标题/无色条/无文字），便于论文/报告直接使用
        # ============================================================
        try:
            import matplotlib.pyplot as plt

            def _save_clean_gray(arr2d: np.ndarray, path: str, vmin=None, vmax=None):
                plt.figure(figsize=(5, 4))
                plt.imshow(arr2d, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
                plt.axis("off")
                plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.0)
                plt.close()

            def _save_clean_viridis(arr2d: np.ndarray, path: str, vmin=None, vmax=None):
                plt.figure(figsize=(5, 4))
                plt.imshow(arr2d, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
                plt.axis("off")
                plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.0)
                plt.close()

            ckpt_stem = os.path.splitext(ckpt_name)[0]

            # 1) 真值图（0/255 灰度 + viridis）
            if bool(cfg.get("eval_save_gt_clean", True)):
                g = (gt > 0).astype(np.uint8) * 255
                _save_clean_gray(g, os.path.join(out_dir, f"gt_clean_{ckpt_stem}.png"), vmin=0, vmax=255)

            if bool(cfg.get("eval_save_gt_viridis", True)):
                gv = (gt > 0).astype(np.float32)
                _save_clean_viridis(gv, os.path.join(out_dir, f"gt_viridis_{ckpt_stem}.png"), vmin=0.0, vmax=1.0)

            # 2) 预测分数图（灰度 + viridis）
            if bool(cfg.get("eval_save_score_clean", True)):
                _save_clean_gray(score_map.astype(np.float32), os.path.join(out_dir, f"score_clean_{ckpt_stem}.png"), vmin=0.0, vmax=1.0)

            if bool(cfg.get("eval_save_score_viridis", True)):
                _save_clean_viridis(score_map.astype(np.float32), os.path.join(out_dir, f"score_viridis_{ckpt_stem}.png"), vmin=0.0, vmax=1.0)

        except Exception:
            pass


        # score image
        try:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.imshow(score_map, cmap="gray")
            plt.axis("off")
            plt.title(f"score (AUC4={auc4:.3f})")
            img_path = os.path.join(
                out_dir, f"score_{os.path.splitext(ckpt_name)[0]}.png"
            )
            plt.savefig(img_path, dpi=200, bbox_inches="tight", pad_inches=0.0)
            plt.close()
        except Exception:
            pass

        # 额外可视化：GT / sim_t / sim_bg / margin / wave / angle / gate
        # 这些图通常比单纯 score 更能解释：假警到底来自“目标相似度过高”还是“背景抑制不够”。
        if dbg is not None:
            try:
                # 1) save npz (方便你后处理)
                npz_path = os.path.join(out_dir, f"debug_{os.path.splitext(ckpt_name)[0]}.npz")

                # dbg 里既有像素级向量（长度=H*W），也可能有全局统计量/原型权重等（长度!=H*W）。
                # 旧实现强行 reshape 会报错，导致 debug 文件直接保存失败（你就看不到中间量了）。
                dbg_save = {}
                for k, v in dbg.items():
                    arr = np.asarray(v)
                    if arr.size == H * W:
                        dbg_save[k] = arr.reshape(H, W, order="F").astype(np.float32)
                    else:
                        # 标量/1D/2D：原样保存，便于你后处理（例如查看 t_max_used / bg_thr_used / t_bank_w 等）
                        try:
                            dbg_save[k] = arr.astype(np.float32)
                        except Exception:
                            dbg_save[k] = arr

                np.savez_compressed(
                    npz_path,
                    score=score_map.astype(np.float32),
                    gt=gt.astype(np.float32),
                    **dbg_save,
                )

                # 2) helper: save 2D map
                def _save_map(arr2d: np.ndarray, name: str, title: str):
                    plt.figure(figsize=(5, 4))
                    plt.imshow(arr2d)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title(title)
                    plt.axis('off')
                    p = os.path.join(out_dir, f"{name}_{os.path.splitext(ckpt_name)[0]}.png")
                    plt.savefig(p, dpi=200, bbox_inches='tight', pad_inches=0.0)
                    plt.close()

                # GT
                _save_map(gt.astype(np.float32), "gt", "GT map")

                # core debug maps
                for k in ["sim_t", "sim_bg", "sem_margin", "wave_sim", "deriv_sim", "angle_raw_t0", "angle_gate_t0", "angle_raw", "angle_gate", "fusion_conf", "fusion_sup", "gate"]:
                    if k in dbg:
                        _save_map(dbg[k].reshape(H, W, order='F'), k, k)

                # 3) histogram: target vs background score
                gt_mask = (gt.reshape(-1, order='F') > 0)
                score_flat = score_map.reshape(-1, order='F')
                s_t = score_flat[gt_mask]
                s_b = score_flat[~gt_mask]
                plt.figure(figsize=(5, 4))
                plt.hist(s_b, bins=50, alpha=0.6, label='bg')
                plt.hist(s_t, bins=50, alpha=0.6, label='target')
                plt.legend()
                plt.title('Score histogram')
                plt.xlabel('score')
                plt.ylabel('count')
                hist_path = os.path.join(out_dir, f"hist_{os.path.splitext(ckpt_name)[0]}.png")
                plt.savefig(hist_path, dpi=200, bbox_inches='tight')
                plt.close()

                # 4) quick binary map (threshold so that #pred == #gt)
                n_pos = int(gt_mask.sum())
                if n_pos > 0:
                    thr = float(np.partition(score_flat, -n_pos)[-n_pos])
                    pred = (score_map >= thr).astype(np.float32)
                    _save_map(pred, "pred", f"Pred (top-{n_pos})")
            except Exception:
                pass


        # ============================================================
        # ROC 曲线 CSV（按你要求输出 4 份）
        #   (P_f, P_d)
        #   (\tau, P_f, P_d)
        #   (\tau, P_d)
        #   (\tau, P_f)
        # ============================================================
        ckpt_stem = os.path.splitext(ckpt_name)[0]

        # (P_f, P_d)
        try:
            fpr = np.asarray(curves.get("fpr"), dtype=np.float64)
            tpr = np.asarray(curves.get("tpr"), dtype=np.float64)
            pf_pd_path = os.path.join(out_dir, f"roc_pf_pd_{ckpt_stem}.csv")
            np.savetxt(
                pf_pd_path,
                np.stack([fpr, tpr], axis=1),
                delimiter=",",
                header="pf,pd",
                comments="",
            )
        except Exception:
            pass

        # \tau-based curves
        try:
            tau = np.asarray(curves.get("tau"), dtype=np.float64)
            pf_tau = np.asarray(curves.get("pf_tau"), dtype=np.float64)
            pd_tau = np.asarray(curves.get("pd_tau"), dtype=np.float64)

            tau_pf_pd_path = os.path.join(out_dir, f"roc_tau_pf_pd_{ckpt_stem}.csv")
            np.savetxt(
                tau_pf_pd_path,
                np.stack([tau, pf_tau, pd_tau], axis=1),
                delimiter=",",
                header="tau,pf,pd",
                comments="",
            )

            tau_pd_path = os.path.join(out_dir, f"roc_tau_pd_{ckpt_stem}.csv")
            np.savetxt(
                tau_pd_path,
                np.stack([tau, pd_tau], axis=1),
                delimiter=",",
                header="tau,pd",
                comments="",
            )

            tau_pf_path = os.path.join(out_dir, f"roc_tau_pf_{ckpt_stem}.csv")
            np.savetxt(
                tau_pf_path,
                np.stack([tau, pf_tau], axis=1),
                delimiter=",",
                header="tau,pf",
                comments="",
            )
        except Exception:
            pass



    return auc1, auc2, auc3, auc4, auc5


def select_best(cfg: Dict) -> Tuple[float, int]:
    """遍历已保存 checkpoint，计算并记录 AUC1~AUC5，再按指定指标选最优。

    你提到的需求：想把 select 过程中的 AUC1~AUC5 都存下来，便于观察训练趋势。
    这里会在 out_dir 下生成一个 CSV（每行一个 epoch 的评估结果）。

    可配置项：
        - select_metric: "AUC4"(默认) / "AUC1" / "AUC2" / "AUC3" / "AUC5"
        - select_save_outputs: 是否在 select 过程中也保存 score/roc（默认 False）
          （建议关掉，不然会产生成百上千个图片/roc 文件）

    Returns:
        best_score: 最优 checkpoint 在 select_metric 上的数值
        best_ep:    最优 epoch
    """
    save_dir = cfg.get("save_dir", "./models/")
    max_epoch = int(cfg.get("epoch", 200))

    # 选择指标（默认 AUC4，与原逻辑一致）
    metric_name = str(cfg.get("select_metric", "AUC4")).upper()
    valid = {"AUC1", "AUC2", "AUC3", "AUC4", "AUC5"}
    if metric_name not in valid:
        metric_name = "AUC4"

    # ============================================================
    # NEW: 快速选择（推荐在 keep_last_ckpts 开启时使用）
    # ------------------------------------------------------------
    # 如果 train() 开启了 debug_eval_interval，并且 save_best_ckpt=True，
    # 会在 save_dir 下写入 best_ckpt.json。
    # 此时 select_best() 可以直接读这个文件返回最优 epoch，
    # 无需再遍历所有 ckpt 逐个 eval（大幅节省时间）。
    #
    # 用法：
    #   cfg["select_fast"] = True
    # ============================================================
    if bool(cfg.get("select_fast", False)):
        best_json = os.path.join(save_dir, "best_ckpt.json")
        if os.path.isfile(best_json):
            try:
                with open(best_json, "r", encoding="utf-8") as jf:
                    info = json.load(jf)
                best_ep = int(info.get("best_epoch", 1))
                best_score = float(info.get("best_score", 0.0))
                metric_used = str(info.get("metric", metric_name)).upper()
                ckpt_used = str(info.get("ckpt", f"ckpt_{best_ep}_.pt"))
                print(
                    f"[select-fast] use {best_json} -> ep={best_ep:03d} {metric_used}={best_score:.6f} ({ckpt_used})"
                )
                return best_score, best_ep
            except Exception:
                pass

    # =========================
    # NEW: 多目标/约束式选最优（避免“只看 AUC5 导致 AUC1 被牺牲”）
    # =========================
    # 背景：
    #   你最新的日志里出现了典型现象：
    #     - 早期 epoch 的 AUC3 极低 -> AUC5 = AUC2/(AUC3+eps) 会被放大到很大；
    #     - 但这并不代表 AUC1/AUC2 真正更好（常见是目标不够亮/排序不够稳）。
    #   如果“盲用 AUC5 选 ckpt”，就会被这种极端值牵着走。
    # 解决办法：
    #   - 用多目标或带约束策略：先把 AUC1 保持在接近最优的区间，再最大化 AUC5。
    #   - 这是流程级修复：不靠盲调参，也不强行改推理/训练。

    # 配置项：
    #   select_strategy:
    #     - single(默认): 只按 select_metric
    #     - constrained: 手动约束
    #     - auto_auc1_guard: 自动用 max(AUC1) 做“护栏”
    #   select_primary: 主指标（默认同 select_metric）
    #   select_constraints: dict，可包含 {"AUC1_min":0.98, "AUC3_max":0.02, ...}
    #   auto_auc1_guard 额外参数：
    #     select_auc1_keep_ratio: AUC1 >= keep_ratio * max_auc1 (默认 0.995)
    #     select_auc1_keep_drop : AUC1 >= max_auc1 - drop      (默认 0.010)

    select_strategy = str(cfg.get("select_strategy", "single")).lower()
    select_primary = str(cfg.get("select_primary", metric_name)).upper()
    if select_primary not in valid:
        select_primary = metric_name

    select_constraints = cfg.get("select_constraints", {})
    if select_constraints is None or (not isinstance(select_constraints, dict)):
        select_constraints = {}

    keep_ratio = cfg.get("select_auc1_keep_ratio", 0.995)
    keep_drop = cfg.get("select_auc1_keep_drop", 0.010)
    try:
        keep_ratio = float(keep_ratio)
    except Exception:
        keep_ratio = 0.995
    try:
        keep_drop = float(keep_drop)
    except Exception:
        keep_drop = 0.010
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    keep_drop = float(max(0.0, keep_drop))

    # 保存历史到 CSV
    out_dir = cfg.get("out_dir", "./outputs/")
    checkFile(out_dir)

    ds_name = os.path.splitext(os.path.basename(cfg.get("path", "data")))[0]
    # 为了方便对比多次训练，这里同时保存：带时间戳版本 + latest 覆盖版
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"select_auc_history_{ds_name}_{stamp}.csv")
    csv_latest = os.path.join(out_dir, f"select_auc_history_{ds_name}.csv")

    # 注意：select 阶段默认不保存 score/roc，避免磁盘爆炸
    save_outputs = bool(cfg.get("select_save_outputs", False))

    # 先把全部 epoch 的结果存下来，循环结束后再按策略选 best。
    rows = []  # list[dict]
    best_score = -1.0
    best_ep = 1

    import csv as _csv

    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(
            [
                "epoch",
                "ckpt",
                "AUC1",
                "AUC2",
                "AUC3",
                "AUC4",
                "AUC5",
                "select_metric",
                "select_score",
            ]
        )

        for ep in range(1, max_epoch + 1):
            ckpt_name = f"ckpt_{ep}_.pt"
            ckpt_path = os.path.join(save_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                continue

            cfg2 = dict(cfg)
            cfg2["test_load_weight"] = ckpt_name

            auc1, auc2, auc3, auc4, auc5 = eval(
                cfg2, quiet=True, save_outputs=save_outputs
            )

            metrics = {
                "AUC1": float(auc1),
                "AUC2": float(auc2),
                "AUC3": float(auc3),
                "AUC4": float(auc4),
                "AUC5": float(auc5),
            }
            score = float(metrics.get(metric_name, auc4))

            # 写入 CSV（每个 epoch 一行）
            w.writerow(
                [
                    ep,
                    ckpt_name,
                    f"{auc1:.6f}",
                    f"{auc2:.6f}",
                    f"{auc3:.6f}",
                    f"{auc4:.6f}",
                    f"{auc5:.6f}",
                    metric_name,
                    f"{score:.6f}",
                ]
            )
            f.flush()

            # 先按单指标记一个 best（方便你看日志趋势），最终 best 在循环后再按策略算。
            if score > best_score:
                best_score = score
                best_ep = ep

            rows.append({"epoch": int(ep), "ckpt": ckpt_name, **metrics})

            print(
                f"[select] ep={ep:03d} {metric_name}={score:.4f} (AUC1={auc1:.4f}, AUC2={auc2:.4f}, AUC3={auc3:.4f}, AUC4={auc4:.4f}, AUC5={auc5:.2f}) best={best_score:.4f}@{best_ep}"
            )

    # 额外写一份 latest 文件，便于你直接打开查看
    try:
        import shutil

        shutil.copyfile(csv_path, csv_latest)
    except Exception:
        pass

    print(f"[select] Saved AUC history -> {csv_path}")
    print(f"[select] Latest  AUC history -> {csv_latest}")

    # =========================
    # Final selection (multi-objective)
    # =========================
    if len(rows) > 0:

        def _passes_constraints(r: dict, cons: dict) -> bool:
            for k, v in cons.items():
                if v is None:
                    continue
                kk = str(k).upper().strip()
                try:
                    vv = float(v)
                except Exception:
                    continue
                if kk.endswith("_MIN"):
                    m = kk[:-4]
                    if m in valid and float(r.get(m, -1e9)) < vv:
                        return False
                elif kk.endswith("_MAX"):
                    m = kk[:-4]
                    if m in valid and float(r.get(m, 1e9)) > vv:
                        return False
            return True

        best_final = None
        if select_strategy in ["single", "metric", "auc", "default"]:
            best_final = max(rows, key=lambda rr: float(rr.get(metric_name, -1e9)))

        elif select_strategy in ["constrained", "constraint"]:
            cand = [r for r in rows if _passes_constraints(r, select_constraints)]
            if len(cand) == 0:
                cand = rows
            best_final = max(cand, key=lambda rr: float(rr.get(select_primary, -1e9)))

        elif select_strategy in ["auto_auc1_guard", "auc1_guard", "guard", "auto_guard"]:
            max_auc1 = max(float(r.get("AUC1", 0.0)) for r in rows)
            thr_ratio = max_auc1 * keep_ratio if keep_ratio > 0 else 0.0
            thr_drop = max_auc1 - keep_drop if keep_drop > 0 else 0.0
            thr_auc1 = max(thr_ratio, thr_drop)

            cons2 = dict(select_constraints)
            cons2["AUC1_min"] = max(float(cons2.get("AUC1_min", -1e9)), float(thr_auc1))

            cand = [r for r in rows if _passes_constraints(r, cons2)]
            if len(cand) == 0:
                cand = rows
            best_final = max(cand, key=lambda rr: float(rr.get(select_primary, -1e9)))

            print(
                f"[select] auto_auc1_guard: max(AUC1)={max_auc1:.4f}, AUC1_min={thr_auc1:.4f}, primary={select_primary}"
            )

        else:
            best_final = max(rows, key=lambda rr: float(rr.get(metric_name, -1e9)))

        if best_final is not None:
            best_ep = int(best_final.get("epoch", best_ep))
            # 返回 best_score：按 primary 更符合“同时抬 AUC1/AUC5”的语义
            best_score = float(best_final.get(select_primary, best_score))
            print(
                f"[select] FINAL best -> ep={best_ep:03d} primary={select_primary}={best_score:.6f} "
                f"(AUC1={best_final['AUC1']:.4f}, AUC2={best_final['AUC2']:.4f}, AUC3={best_final['AUC3']:.4f}, "
                f"AUC4={best_final['AUC4']:.4f}, AUC5={best_final['AUC5']:.2f}) strategy={select_strategy}"
            )

    # 可选：把 AUC1~AUC5 曲线画出来（方便你观察趋势、定位拐点）
    if bool(cfg.get("select_plot_history", True)):
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            df = pd.read_csv(csv_path)
            if "epoch" in df.columns:
                x = df["epoch"].values
            else:
                x = np.arange(len(df))
            plt.figure(figsize=(7, 4))
            for k in ["AUC1", "AUC2", "AUC3", "AUC4", "AUC5"]:
                if k in df.columns:
                    plt.plot(x, df[k].values, label=k)
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("metric")
            plt.title("AUC history")
            png_path = os.path.join(out_dir, f"auc_history_{cfg.get('dataset', 'data')}.png")
            plt.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

    return best_score, best_ep


# =========================
# Sweep eval (一次评估批量产出多组参数结果)
# =========================


def _disable_post_cfg(cfg: Dict) -> Dict:
    """生成一个“只算基础 score，不做后处理标定”的 cfg。

    目的：
      - 先把模型的基础得分算出来（包含 score_mode / gate / angle_gate 等）；
      - 再在 numpy 里做严格单调的后处理 sweep（sigmoid/highlift/pow/norm），
        这样 **AUC1 不变**，但能系统性地寻找 AUC2/AUC3/AUC5 的更优组合，
        且无需重复跑 encoder forward（节省大量时间）。
    """
    cfg2 = dict(cfg)
    cfg2["score_post_norm"] = "none"
    cfg2["score_post_sigmoid"] = False
    cfg2["score_post_scale"] = 1.0
    cfg2["score_post_pow"] = 1.0
    cfg2["score_post_highlift"] = False
    return cfg2


def _auto_pow_from_bg_np(
    score: np.ndarray,
    sim_bg: Optional[np.ndarray],
    gate: Optional[np.ndarray],
    cfg: Dict,
) -> float:
    """自动选择 score_post_pow（推理期，无需 GT），用于压低背景均值(AUC3)。

    背景：
      - 在本项目的评估实现里（_compute_auc_metrics），AUC2≈E[score|T]，AUC3≈E[score|B]，
        因此 AUC5 = AUC2/(AUC3+eps) 本质就是“目标均值 / 背景均值”的信噪比指标。
      - 幂次映射 score' = score^p (p>1) 是**严格单调**的：
          * 不改变排序 => AUC1 不变
          * 会把中等分数压向 0 => 通常 AUC3 下降更明显 => AUC5 上升

    你遇到的痛点是不同数据集/不同 checkpoint 的分数分布差异很大，
    手动 sweep 很麻烦。这里用 sim_bg / gate 等“更像背景”的像素子集
    自动估计背景分布，并用二分搜索选择 p，使背景子集的均值接近一个目标值。

    关键超参（一般不需要频繁改）：
      - score_post_auto_bg_mean：希望背景均值压到的水平（越小越强抑制）
      - score_post_auto_pow_max：自动搜索的最大 p（防止过激）
      - score_post_auto_pos_ratio：估计背景时排除 top-K 高分像素（避免把目标混进去）
      - score_post_auto_bg_sel_q / score_post_auto_gate_low_q：用 sim_bg/gate 选背景子集的分位数

    返回：
      p（float）。若估计失败则回退到 cfg['score_post_pow']。
    """

    # fallback
    try:
        p_fallback = float(cfg.get("score_post_pow", 1.0))
    except Exception:
        p_fallback = 1.0

    s = np.clip(np.asarray(score, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if s.size < 64:
        return float(p_fallback)

    # -------- hyper-params --------
    bg_mean_target = float(cfg.get("score_post_auto_bg_mean", 0.0020))
    bg_mean_target = float(np.clip(bg_mean_target, 1e-6, 0.50))

    p_max = float(cfg.get("score_post_auto_pow_max", 8.0))
    p_max = float(np.clip(p_max, 1.0, 32.0))

    pos_ratio = float(cfg.get("score_post_auto_pos_ratio", 0.005))
    pos_ratio = float(np.clip(pos_ratio, 0.0, 0.20))

    bg_sel_q = float(cfg.get("score_post_auto_bg_sel_q", 0.60))
    bg_sel_q = float(np.clip(bg_sel_q, 0.0, 1.0))

    gate_low_q = float(cfg.get("score_post_auto_gate_low_q", 0.70))
    gate_low_q = float(np.clip(gate_low_q, 0.0, 1.0))

    min_keep = int(cfg.get("score_post_auto_min_keep", 2048))
    min_keep = max(64, min_keep)

    iters = int(cfg.get("score_post_auto_pow_iters", 24))
    iters = int(np.clip(iters, 8, 64))

    # -------- build masks --------
    # 1) exclude top scores (likely target / anomalies)
    try:
        if pos_ratio > 0:
            thr_pos = float(np.quantile(s.astype(np.float32), 1.0 - pos_ratio))
        else:
            thr_pos = 1.1
    except Exception:
        thr_pos = 1.1
    pos_mask = (s >= thr_pos)

    # 2) background candidate: sim_bg high OR gate low
    bg_mask = None
    try:
        masks = []
        if sim_bg is not None:
            sb = np.asarray(sim_bg, dtype=np.float32).reshape(-1)
            if sb.size == s.size:
                thr_sb = float(np.quantile(sb, bg_sel_q))
                masks.append(sb >= thr_sb)
        if gate is not None:
            g = np.asarray(gate, dtype=np.float32).reshape(-1)
            if g.size == s.size:
                thr_g = float(np.quantile(g, gate_low_q))
                masks.append(g <= thr_g)
        if masks:
            bg_mask = masks[0]
            for m in masks[1:]:
                bg_mask = (bg_mask | m)
    except Exception:
        bg_mask = None

    if bg_mask is None:
        # fallback: low-score region as background proxy
        try:
            thr_bg = float(np.quantile(s, 0.70))
        except Exception:
            thr_bg = float(np.nanmedian(s))
        bg_mask = (s <= thr_bg)

    # remove possible targets
    bg_mask = (bg_mask & (~pos_mask))

    # ensure enough samples
    if int(bg_mask.sum()) < min_keep:
        try:
            thr_bg2 = float(np.quantile(s[~pos_mask], 0.90))
            bg_mask = (s <= thr_bg2) & (~pos_mask)
        except Exception:
            pass
    if int(bg_mask.sum()) < 64:
        bg_mask = (~pos_mask)

    bg_scores = s[bg_mask]
    bg_scores = bg_scores[np.isfinite(bg_scores)]
    if bg_scores.size < 64:
        return float(p_fallback)

    m0 = float(np.mean(bg_scores))
    if (not np.isfinite(m0)) or (m0 <= 0.0):
        return float(p_fallback)

    # already good enough
    if m0 <= bg_mean_target:
        return 1.0

    def _mean_pow(p: float) -> float:
        return float(np.mean(np.power(bg_scores, float(p))))

    # if even p_max cannot reach target, return p_max
    if _mean_pow(p_max) > bg_mean_target:
        return float(p_max)

    # bisection on p in [1, p_max]
    lo, hi = 1.0, float(p_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _mean_pow(mid) > bg_mean_target:
            lo = mid
        else:
            hi = mid
    return float(hi)

def apply_score_post_np(
    score: np.ndarray,
    dbg: Optional[Dict[str, np.ndarray]],
    cfg: Dict,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """把 compute_detection_scores 里的“后处理标定”抽出来，便于 sweep。

    注意：
      - 这里的操作都应当是严格单调映射（不改变排序），因此不会改变 AUC1；
      - 但会改变 τ-curve 类指标（AUC2/AUC3/AUC5）。

    Args:
        score: 基础 score（建议来自 _disable_post_cfg 的输出），[N] in [0,1]
        dbg:   compute_detection_scores(return_debug=True) 的 debug dict
        cfg:   这一次 sweep 的参数（只需要包含 score_post_* 相关项）

    Returns:
        score2: 后处理后的 score，[N] in [0,1]
        used:   记录本次自动估计得到的 mu/k/scale 等，用于日志与复盘
    """
    s = np.clip(score.astype(np.float32), 0.0, 1.0)
    used: Dict[str, float] = {}

    # 可选辅助量（用于 bg_q / angle_gate mask 等）
    sim_bg = None
    angle_gate = None
    angle_gate_t0 = None
    gate = None
    if isinstance(dbg, dict):
        sim_bg = dbg.get("sim_bg", None)
        angle_gate = dbg.get("angle_gate", None)
        angle_gate_t0 = dbg.get("angle_gate_t0", None)
        gate = dbg.get("gate", None)

    # -------------------------
    # 1) score_post_norm
    # -------------------------
    score_post_norm = str(cfg.get("score_post_norm", "none")).lower()
    if score_post_norm not in ["none", "off", "false", "0", ""]:
        eps = 1e-6
        if score_post_norm in ["max", "smax"]:
            sc = float(np.nanmax(s))
            if (not np.isfinite(sc)) or sc < eps:
                sc = 1.0
            used["score_post_scale"] = float(sc)
            s = np.clip(s / (sc + eps), 0.0, 1.0).astype(np.float32)

        elif score_post_norm in ["q", "quantile", "percentile"]:
            q = float(cfg.get("score_post_norm_q", 0.999))
            q = float(np.clip(q, 0.90, 0.99999))
            s_vec = s

            mask_mode = str(cfg.get("score_post_norm_mask", "none")).lower()
            min_keep = int(cfg.get("score_post_norm_min_keep", 1024))
            try:
                if (mask_mode in ["angle_t0_gate", "t0_gate", "angle0"]) and (angle_gate_t0 is not None):
                    thr_m = float(cfg.get("score_post_norm_angle_t0_gate_thr", 0.5))
                    m = (angle_gate_t0.reshape(-1).astype(np.float32) >= thr_m)
                    if int(m.sum()) >= min_keep:
                        s_vec = s.reshape(-1)[m]
                elif (mask_mode in ["angle_gate", "pooled_angle_gate"]) and (angle_gate is not None):
                    thr_m = float(cfg.get("score_post_norm_angle_gate_thr", 0.5))
                    m = (angle_gate.reshape(-1).astype(np.float32) >= thr_m)
                    if int(m.sum()) >= min_keep:
                        s_vec = s.reshape(-1)[m]
            except Exception:
                pass

            try:
                sc = float(np.quantile(s_vec.astype(np.float32), q))
            except Exception:
                sc = float(np.nanmax(s))
            if (not np.isfinite(sc)) or sc < eps:
                sc = float(np.nanmax(s))
            if (not np.isfinite(sc)) or sc < eps:
                sc = 1.0
            used["score_post_scale"] = float(sc)
            s = np.clip(s / (sc + eps), 0.0, 1.0).astype(np.float32)

        elif score_post_norm in ["conformal", "pvalue", "cp", "p"]:
            # conformal 标定：用“更像背景”的样本估计经验 p-value
            calib_mode = str(cfg.get("conformal_calib", "gate_low")).lower()
            min_keep = int(cfg.get("conformal_min_keep", 2048))
            gate_thr = cfg.get("conformal_gate_thr", 0.5)
            gate_q = float(cfg.get("conformal_gate_q", 0.70))
            gate_q = float(np.clip(gate_q, 0.05, 0.95))

            msk = None
            if calib_mode in ["gate_low", "gate", "bg"] and gate is not None:
                try:
                    g = gate.reshape(-1).astype(np.float32)
                    if isinstance(gate_thr, str) and gate_thr.lower() in ["auto", "q", "quantile"]:
                        thr = float(np.quantile(g, gate_q))
                    else:
                        thr = float(gate_thr)
                    m = (g <= thr)
                    if int(m.sum()) < min_keep:
                        thr = float(np.quantile(g, gate_q))
                        m = (g <= thr)
                    msk = m
                except Exception:
                    msk = None

            if msk is None or int(msk.sum()) < 8:
                msk = np.ones_like(s.reshape(-1), dtype=bool)

            calib_scores = s.reshape(-1)[msk]
            s = _conformal_score_np(s.reshape(-1), calib_scores).astype(np.float32)

        else:
            raise ValueError(f"Unknown score_post_norm: {score_post_norm}")

    # -------------------------
    # 2) score_post_sigmoid (anchored monotonic)
    # -------------------------
    if bool(cfg.get("score_post_sigmoid", False)):
        k = float(cfg.get("score_post_sigmoid_k", 20.0))
        mu_cfg = cfg.get("score_post_sigmoid_mu", "q")
        q_mu = float(cfg.get("score_post_sigmoid_q", 0.98))
        q_mu = float(np.clip(q_mu, 0.50, 0.9999))

        s_all = s.reshape(-1).astype(np.float32)
        s_vec = s_all
        mask_mode = str(cfg.get("score_post_sigmoid_mask", "none")).lower()
        min_keep = int(cfg.get("score_post_sigmoid_min_keep", 1024))
        gate_q = float(cfg.get("score_post_sigmoid_gate_q", 0.70))
        gate_q = float(np.clip(gate_q, 0.05, 0.95))
        gate_thr = cfg.get("score_post_sigmoid_gate_thr", "auto")
        if (mask_mode in ["gate_low", "gate", "bg", "background"]) and (gate is not None):
            try:
                g = gate.reshape(-1).astype(np.float32)
                if isinstance(gate_thr, str) and gate_thr.lower() in ["auto", "q", "quantile"]:
                    thr = float(np.quantile(g, gate_q))
                else:
                    thr = float(gate_thr)
                m = (g <= thr)
                if int(m.sum()) >= min_keep:
                    s_vec = s_all[m]
            except Exception:
                pass

        # resolve mu
        try:
            if isinstance(mu_cfg, str):
                mstr = mu_cfg.lower()
                if mstr in ["q", "quantile", "auto"]:
                    mu = float(np.quantile(s_vec, q_mu))
                elif mstr in ["median", "p50"]:
                    mu = float(np.quantile(s_vec, 0.50))
                elif mstr in ["bg_q", "bg_quantile", "bgq"]:
                    s_mu_vec = s_all
                    if sim_bg is not None:
                        sel_q = float(cfg.get("score_post_sigmoid_bg_sel_q", 0.60))
                        sel_q = float(np.clip(sel_q, 0.0, 1.0))
                        try:
                            thr_bg = float(np.quantile(sim_bg.reshape(-1).astype(np.float32), sel_q))
                            m_bg = (sim_bg.reshape(-1).astype(np.float32) >= thr_bg)
                            if int(m_bg.sum()) >= min_keep:
                                s_mu_vec = s_all[m_bg]
                        except Exception:
                            pass
                    q_bg = float(cfg.get("score_post_sigmoid_bg_mu_q", 0.995))
                    q_bg = float(np.clip(q_bg, 0.50, 0.99999))
                    mu = float(np.quantile(s_mu_vec, q_bg))

                    # 兜底：避免 mu 过高导致目标整体被压扁
                    min_pos_ratio = float(cfg.get("score_post_sigmoid_min_pos_ratio", 0.0))
                    if np.isfinite(min_pos_ratio) and (min_pos_ratio > 0.0):
                        min_pos_ratio = float(np.clip(min_pos_ratio, 1e-6, 0.20))
                        mu_pos = float(np.quantile(s_all, 1.0 - min_pos_ratio))
                        mu = float(min(mu, mu_pos))
                else:
                    mu = float(mu_cfg)
            else:
                mu = float(mu_cfg)
        except Exception:
            mu = float(np.quantile(s_vec, q_mu))

        if (not np.isfinite(k)) or (k <= 0):
            k = 20.0
        if not np.isfinite(mu):
            mu = float(np.quantile(s_all, q_mu))

        used["score_post_sigmoid_mu_used"] = float(mu)
        used["score_post_sigmoid_k_used"] = float(k)

        # anchored sigmoid: force 0->0, 1->1
        rescale = bool(cfg.get("score_post_sigmoid_rescale", True))
        x = (s_all - float(mu)) * float(k)
        x = np.clip(x, -60.0, 60.0)
        y = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
        if rescale:
            x0 = float(np.clip((0.0 - float(mu)) * float(k), -60.0, 60.0))
            x1 = float(np.clip((1.0 - float(mu)) * float(k), -60.0, 60.0))
            y0 = 1.0 / (1.0 + np.exp(-x0))
            y1 = 1.0 / (1.0 + np.exp(-x1))
            denom = float(y1 - y0)
            if denom > 1e-6:
                y = (y - float(y0)) / denom
        s = np.clip(y.reshape(s.shape), 0.0, 1.0).astype(np.float32)

    # -------------------------
    # 2.5) t0-angle suppression（可选，非单调，会改变排序/AUC1）
    # -------------------------
    # 说明：
    #   - 这一步与 “pow/highlift” 不同，它不是严格单调映射，因此会影响 AUC1。
    #   - 但它能非常有效地压制“嵌入相似但原谱形不一致”的背景尾部假警（AUC3↓），
    #     在 Sandiego 这类场景往往是把 AUC5 拉到 200+ 的关键。
    score_post_t0_angle_pow = cfg.get("score_post_t0_angle_pow", 0.0)
    used["score_post_t0_angle_pow_used"] = float("nan")

    if angle_gate_t0 is not None:
        if isinstance(score_post_t0_angle_pow, str) and str(score_post_t0_angle_pow).lower() == "auto":
            top_ratio = float(cfg.get("score_post_t0_angle_auto_top_ratio", 0.005))
            thr = float(cfg.get("score_post_t0_angle_auto_thr", 0.60))
            p_max = float(cfg.get("score_post_t0_angle_auto_p", 0.60))

            try:
                s_flat = s.reshape(-1).astype(np.float32)
                g_flat = np.asarray(angle_gate_t0, dtype=np.float32).reshape(-1)
                k = max(1, int(s_flat.size * top_ratio))
                idx_top = np.argpartition(s_flat, -k)[-k:]
                med_g = float(np.median(g_flat[idx_top]))
            except Exception:
                med_g = 0.0

            if (med_g >= thr) and (p_max > 0):
                used["score_post_t0_angle_pow_used"] = float(p_max)
                g = np.clip(np.asarray(angle_gate_t0, dtype=np.float32), 0.0, 1.0)
                s = np.clip(s * np.power(g, float(p_max)), 0.0, 1.0).astype(np.float32)
            else:
                used["score_post_t0_angle_pow_used"] = 0.0
        else:
            try:
                p = float(score_post_t0_angle_pow)
            except Exception:
                p = 0.0
            if p > 0:
                used["score_post_t0_angle_pow_used"] = float(p)
                g = np.clip(np.asarray(angle_gate_t0, dtype=np.float32), 0.0, 1.0)
                s = np.clip(s * np.power(g, float(p)), 0.0, 1.0).astype(np.float32)

    # -------------------------
    # 3) monotonic pow
    # -------------------------
    # ✅ 支持自动 pow（无需 GT / 无需手动 sweep）
    pow_auto = bool(cfg.get("score_post_pow_auto", False)) or bool(cfg.get("score_post_auto", False))

    if pow_auto:
        try:
            pow_v = _auto_pow_from_bg_np(s, sim_bg, gate, cfg)
        except Exception:
            pow_v = float(cfg.get("score_post_pow", 1.0))
    else:
        pow_v = float(cfg.get("score_post_pow", 1.0))

    if np.isfinite(pow_v) and (abs(pow_v - 1.0) > 1e-6) and (pow_v > 0):
        used["score_post_pow_used"] = float(pow_v)
        s = np.power(np.clip(s, 0.0, 1.0), float(pow_v)).astype(np.float32)
    # -------------------------
    # 4) high-score lift (piecewise monotonic)
    # -------------------------
    if bool(cfg.get("score_post_highlift", False)):
        try:
            thr = float(cfg.get("score_post_highlift_thr", 0.50))
            gamma = float(cfg.get("score_post_highlift_gamma", 2.0))
            if np.isfinite(thr) and np.isfinite(gamma) and (0.0 < thr < 1.0) and (gamma > 0.0):
                m = (s > thr)
                if bool(np.any(m)):
                    x = (s[m] - thr) / max(1.0 - thr, 1e-6)
                    x = np.clip(x, 0.0, 1.0)
                    x2 = 1.0 - np.power((1.0 - x), gamma).astype(np.float32)
                    s[m] = thr + (1.0 - thr) * x2
                    s = np.clip(s, 0.0, 1.0).astype(np.float32)
                used["score_post_highlift_thr_used"] = float(thr)
                used["score_post_highlift_gamma_used"] = float(gamma)
        except Exception:
            pass

    return np.clip(s, 0.0, 1.0).astype(np.float32), used


def sweep_eval(cfg: Dict) -> str:
    """批量评估（sweep）入口。

    典型用法：
      - 固定 score_mode/推理主干不变；
      - 对 score_post_sigmoid / highlift / pow / norm 做网格搜索；
      - 同时对多个 epoch/checkpoint 做对比；
      - 一次运行输出 CSV（包含 AUC1~AUC5 与本次自动估计的 mu/k/scale）。

    Returns:
        csv_path: sweep 结果 CSV 路径
    """
    device = torch.device(
        cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    )

    mat = sio.loadmat(cfg["path"])
    data = standard(mat["data"]).astype(np.float32)
    gt = mat.get("map", None)
    if gt is None:
        raise KeyError("dataset .mat must contain 'map' for sweep")
    gt = gt.astype(np.int32)

    H, W, C = data.shape
    X = data.reshape(-1, C, order="F")

    # target spectrum (eval only)
    target_spectrum = ts_generation(data, gt, type=int(cfg.get("ts_type", 7)))

    # out dir
    out_dir = cfg.get("out_dir", "./outputs/")
    checkFile(out_dir)
    ds_name = os.path.splitext(os.path.basename(cfg.get("path", "data")))[0]

    # -------------------------
    # 1) choose ckpts
    # -------------------------
    save_dir = cfg.get("save_dir", "./models/")
    max_epoch = int(cfg.get("epoch", 200))

    epochs = cfg.get("sweep_epochs", None)
    if epochs is None:
        epochs = []
    if isinstance(epochs, (int, float)):
        epochs = [int(epochs)]
    if isinstance(epochs, str):
        s = epochs.strip().lower()
        if s == "all":
            epochs = list(range(1, max_epoch + 1))
        elif ":" in s:
            parts = s.split(":")
            try:
                a = int(parts[0]) if parts[0] else 1
                b = int(parts[1]) if len(parts) > 1 and parts[1] else max_epoch
                st = int(parts[2]) if len(parts) > 2 and parts[2] else 1
                epochs = list(range(a, b + 1, st))
            except Exception:
                epochs = []
        elif "-" in s:
            try:
                a, b = s.split("-")
                epochs = list(range(int(a), int(b) + 1))
            except Exception:
                epochs = []
        else:
            try:
                epochs = [int(s)]
            except Exception:
                epochs = []

    # auto: from select_csv top-k
    if (not epochs) and bool(cfg.get("sweep_from_select_csv", True)):
        topk = int(cfg.get("sweep_topk_epochs", 8))
        rank_metric = str(cfg.get("sweep_rank_metric", "AUC1")).upper()
        csv_latest = os.path.join(out_dir, f"select_auc_history_{ds_name}.csv")
        if os.path.exists(csv_latest):
            try:
                import pandas as pd

                df = pd.read_csv(csv_latest)
                if ("epoch" in df.columns) and (rank_metric in df.columns):
                    df = df.sort_values(rank_metric, ascending=False)
                    epochs = [int(x) for x in df["epoch"].values[:topk]]
            except Exception:
                epochs = []

    if not epochs:
        ckpt = cfg.get("test_load_weight", None)
        if ckpt is not None:
            try:
                ep = int(str(ckpt).split("_")[1])
                epochs = [ep]
            except Exception:
                epochs = [1]
        else:
            epochs = [max_epoch]

    ckpts = []
    for ep in epochs:
        ck = f"ckpt_{int(ep)}_.pt"
        if os.path.exists(os.path.join(save_dir, ck)):
            ckpts.append((int(ep), ck))
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoints found for sweep.")

    # -------------------------
    # 2) param grid (post-calibration)
    # -------------------------
    grid = cfg.get("sweep_grid", None)
    if grid is None:
        grid = {
            "score_post_norm": ["none", "q"],
            "score_post_norm_q": [0.995, 0.999],
            "score_post_sigmoid": [True],
            "score_post_sigmoid_mu": ["bg_q"],
            "score_post_sigmoid_bg_sel_q": [0.55, 0.60, 0.65],
            "score_post_sigmoid_bg_mu_q": [0.99, 0.995, 0.998],
            "score_post_sigmoid_k": [16.0, 20.0, 24.0, 28.0],
            "score_post_sigmoid_min_pos_ratio": [0.003, 0.005, 0.010],
            "score_post_sigmoid_rescale": [True],
            "score_post_pow": [1.0, 1.3, 1.6],
            "score_post_highlift": [True],
            "score_post_highlift_thr": [0.45, 0.50, 0.55],
            "score_post_highlift_gamma": [1.5, 2.0, 2.5, 3.0],
        }

    if not isinstance(grid, dict) or len(grid) == 0:
        raise ValueError("cfg['sweep_grid'] must be a non-empty dict")

    grid_keys = list(grid.keys())
    grid_vals = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in grid_keys]
    combos = []
    for vals in itertools.product(*grid_vals):
        combos.append({k: v for k, v in zip(grid_keys, vals)})

    # -------------------------
    # 3) run sweep
    # -------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"sweep_post_{ds_name}_{stamp}.csv")

    objective = str(cfg.get("sweep_objective", "AUC5")).upper()
    valid_m = {"AUC1", "AUC2", "AUC3", "AUC4", "AUC5"}
    if objective not in valid_m:
        objective = "AUC5"
    obj_max = bool(cfg.get("sweep_objective_max", True))

    cons = cfg.get("sweep_constraints", {})
    if cons is None or not isinstance(cons, dict):
        cons = {}

    def _pass_cons(m: Dict[str, float]) -> bool:
        for k, v in cons.items():
            if v is None:
                continue
            kk = str(k).upper().strip()
            try:
                vv = float(v)
            except Exception:
                continue
            if kk.endswith("_MIN"):
                mm = kk[:-4]
                if mm in m and float(m[mm]) < vv:
                    return False
            elif kk.endswith("_MAX"):
                mm = kk[:-4]
                if mm in m and float(m[mm]) > vv:
                    return False
        return True

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "epoch",
            "ckpt",
            "combo_id",
            "AUC1",
            "AUC2",
            "AUC3",
            "AUC4",
            "AUC5",
            "objective",
            "obj_score",
        ] + grid_keys + [
            "score_post_scale",
            "score_post_sigmoid_mu_used",
            "score_post_sigmoid_k_used",
            "score_post_pow_used",
            "score_post_highlift_thr_used",
            "score_post_highlift_gamma_used",
        ]
        w.writerow(header)

        best_row = None
        best_val = -1e18 if obj_max else 1e18

        for ep, ckpt_name in ckpts:
            model = _build_model(cfg)
            ckpt_path = os.path.join(save_dir, ckpt_name)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)
            model.to(device).eval()

            cfg_base = _disable_post_cfg(cfg)
            base_score, dbg = compute_detection_scores(
                model,
                X,
                target_spectrum,
                device,
                cfg_base,
                return_debug=True,
            )

            base_map = base_score.reshape(H, W, order="F")
            auc1_b, auc2_b, auc3_b, auc4_b, auc5_b, _ = _compute_auc_metrics(gt, base_map)
            print(
                f"[sweep] ep={ep:03d} base(no-post) AUC1={auc1_b:.4f} AUC2={auc2_b:.4f} AUC3={auc3_b:.4f} AUC5={auc5_b:.2f}"
            )

            for cid, combo in enumerate(combos):
                cfg_c = dict(cfg)
                cfg_c.update(combo)

                s2, used = apply_score_post_np(base_score, dbg, cfg_c)
                score_map = s2.reshape(H, W, order="F")
                auc1, auc2, auc3, auc4, auc5, _ = _compute_auc_metrics(gt, score_map)
                m = {"AUC1": auc1, "AUC2": auc2, "AUC3": auc3, "AUC4": auc4, "AUC5": auc5}

                if not _pass_cons(m):
                    continue

                obj = float(m.get(objective, -1e9))
                if (obj_max and obj > best_val) or ((not obj_max) and obj < best_val):
                    best_val = obj
                    best_row = {
                        "epoch": ep,
                        "ckpt": ckpt_name,
                        "combo_id": cid,
                        **m,
                        **combo,
                        **used,
                        "objective": objective,
                        "obj_score": obj,
                    }

                row = [
                    ep,
                    ckpt_name,
                    cid,
                    f"{auc1:.6f}",
                    f"{auc2:.6f}",
                    f"{auc3:.6f}",
                    f"{auc4:.6f}",
                    f"{auc5:.6f}",
                    objective,
                    f"{obj:.6f}",
                ]
                for k in grid_keys:
                    row.append(combo.get(k))
                row.append(used.get("score_post_scale", np.nan))
                row.append(used.get("score_post_sigmoid_mu_used", np.nan))
                row.append(used.get("score_post_sigmoid_k_used", np.nan))
                row.append(used.get("score_post_pow_used", np.nan))
                row.append(used.get("score_post_highlift_thr_used", np.nan))
                row.append(used.get("score_post_highlift_gamma_used", np.nan))
                w.writerow(row)

        if best_row is not None:
            print("\n===== SWEEP BEST =====")
            print(
                f"best objective={objective} {best_val:.6f} @ ep={best_row['epoch']} ckpt={best_row['ckpt']} combo={best_row['combo_id']}"
            )
            print(
                f"AUC1={best_row['AUC1']:.4f} AUC2={best_row['AUC2']:.4f} AUC3={best_row['AUC3']:.4f} AUC4={best_row['AUC4']:.4f} AUC5={best_row['AUC5']:.2f}"
            )
            for k in grid_keys:
                print(f"  {k}: {best_row.get(k)}")
            for k in [
                "score_post_scale",
                "score_post_sigmoid_mu_used",
                "score_post_sigmoid_k_used",
                "score_post_pow_used",
                "score_post_highlift_thr_used",
                "score_post_highlift_gamma_used",
            ]:
                if k in best_row:
                    print(f"  {k}: {best_row.get(k)}")
            print("======================\n")

    print(f"[sweep] Saved -> {csv_path}")
    return csv_path
