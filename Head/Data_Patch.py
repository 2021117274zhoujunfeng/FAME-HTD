# -*- coding: utf-8 -*-
"""Data_Patch.py

HTD-Mamba 改良版：自监督训练数据集（pixel -> window/patch）。

每个像素返回 6 个张量：
  1) center_v1       : 中心像素（弱增强）          [1, B]
  2) center_v2       : 中心像素（强增强）          [1, B]
  3) coded_vector_v2 : patch 上下文编码（SESA 视角，默认由 RAW patch 生成）  [1, B]
  4) hard_neg_v2     : 同 patch 的 hard-negative  [1, B]
  5) center_raw      : 无增强中心像素              [1, B]
  6) ctx_angle_raw   : RAW 域 center 与 coded 的**谱角距离**(radian)
                       用于训练时门控异质 patch（角度越小越相似）。

为什么改成“谱角/相关系数式余弦”？
- 你的数据在 Data_Patch.standard() 后是 [0,1] 且全正。
  这种情况下直接 cosine similarity 往往会普遍偏高，导致：
  - ctx_sim 门控几乎全为 1（等于没门控，异质 patch 也被强行拉近）
  - density 权重几乎全为 1（原型被稀有目标/异常污染）

这里统一用 **相关系数式余弦 (correlation cosine)** 来算相似度：
    sim(x,y) = cos( (x-mean(x)), (y-mean(y)) )
再把 sim 转为谱角： angle = arccos(clamp(sim)).
这样阈值（如 0.22 rad ≈ 12.6°）才有物理意义，也更利于阈值稳定。

注意：
- 这里只改“相似度/门控”的度量方式；输出光谱仍然是 [0,1] 的原始/增强结果。
- patch 编码（SESA 思路）仍然是：对 patch 像素按相似度做 softmax 加权求和。

"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data

from augment import chain_safe, standard


# =========================
# similarity helpers
# =========================

def _row_center(x: np.ndarray) -> np.ndarray:
    """按行减均值（去亮度偏置）。x: [...,B]"""
    return x - x.mean(axis=-1, keepdims=True)


def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def corr_cosine_similarity_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """相关系数式余弦相似度（对每条光谱先减均值再 cosine）。

    Args:
        x: [N,B]
        y: [1,B] or [N,B]

    Returns:
        sim: [N] in [-1,1]
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_c = _row_center(x)
    y_c = _row_center(y)

    x_u = _l2norm(x_c, axis=1)
    y_u = _l2norm(y_c, axis=1)

    # 支持 y 只有 1 行的广播
    sim = np.sum(x_u * y_u, axis=1)
    return np.clip(sim, -1.0, 1.0)




def cosine_similarity_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """普通余弦相似度（不做减均值）。与论文/原版 HTD-Mamba 的 SESA 权重一致。

    Args:
        x: [N,B]
        y: [1,B] or [N,B]

    Returns:
        sim: [N] in [-1,1]
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_u = _l2norm(x, axis=1)
    y_u = _l2norm(y, axis=1)

    sim = np.sum(x_u * y_u, axis=1)
    return np.clip(sim, -1.0, 1.0)

def spectral_angle_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """谱角距离（radian）。角度越小越相似。"""
    sim = corr_cosine_similarity_rows(x, y)
    return np.arccos(sim).astype(np.float32)


# =========================
# patch encode / hard negative
# =========================

def patch_encoded(
    patch: np.ndarray,
    center: np.ndarray,
    tau: float = 1.0,
    sim_mode: str = "corr",
    spatial_sigma: Optional[float] = None,
    topk_ratio: Optional[float] = None,
) -> np.ndarray:
    """SESA-like：用 patch 内像素按“谱相似度”加权，生成中心像素的新视角。

    Args:
        patch:  [P,P,B]
        center: [1,B]
        tau: softmax 温度
        spatial_sigma: 若不为 None，叠加一个空间高斯权重（更偏向中心邻域）
        topk_ratio: 只保留最相似的 top-k（避免把异质光谱混入上下文）

    Returns:
        encoded: [1,B]
    """
    p_h, p_w, b = patch.shape
    patch_unfold = np.reshape(patch, [-1, b], order="F")  # [N,B]

    # 1) 相似度（SESA 权重用的相似度）
    # - cosine : 原版/论文 HTD-Mamba 的做法
    # - corr   : 对每条谱先减均值再 cosine（对亮度/基线更不敏感，但也可能改变邻域权重分布）
    _m = str(sim_mode).lower()
    if _m in ["cos", "cosine", "raw"]:
        sim = cosine_similarity_rows(patch_unfold, center)  # [N]
    else:
        # corr / other：对谱先减均值再 cosine，更稳健但也可能更“平滑”
        sim = corr_cosine_similarity_rows(patch_unfold, center)  # [N]

# 2) 空间权重（可选）
    if spatial_sigma is not None and spatial_sigma > 0:
        ys, xs = np.meshgrid(np.arange(p_h), np.arange(p_w), indexing="ij")
        cy, cx = p_h // 2, p_w // 2
        d2 = (ys - cy) ** 2 + (xs - cx) ** 2
        w_spatial = np.exp(-d2.astype(np.float32) / (2.0 * (float(spatial_sigma) ** 2)))
        w_spatial = np.reshape(w_spatial, [-1], order="F").astype(np.float32)
    else:
        w_spatial = None

    # 3) 温度 softmax
    tau = float(max(tau, 1e-4))
    logits = sim.astype(np.float32) / tau
    # 注：原始 HTD-Mamba 中权重≈exp(sim)/sum exp(sim)，等价于 tau≈1；tau 太小会让编码视角几乎退化为中心像素。

    # 4) Top-k 过滤：只用最相似的部分做上下文
    if topk_ratio is not None:
        q = float(topk_ratio)
        if q > 0:
            N = logits.shape[0]
            if q <= 1.0:
                k = int(max(1, round(N * q)))
            else:
                k = int(min(N, round(q)))
            if k < N:
                idx = np.argpartition(sim, -k)[-k:]
                mask = np.zeros((N,), dtype=bool)
                mask[idx] = True
                logits2 = np.full_like(logits, -1e9, dtype=np.float32)
                logits2[mask] = logits[mask]
                logits = logits2

    logits = logits - np.max(logits)
    w = np.exp(logits).astype(np.float32)

    if w_spatial is not None:
        w = w * w_spatial

    w = w / (np.sum(w) + 1e-8)
    encoded = np.sum((w[:, None]) * patch_unfold.astype(np.float32), axis=0)  # [B]
    return encoded[None, :].astype(np.float32)


def hard_negative_from_patch(
    patch: np.ndarray,
    center: np.ndarray,
    center_hw: Optional[Tuple[int, int]] = None,
    hard_quantile: float = 0.10,
    sim_mode: str = "corr",
    candidate_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """从同一 patch 中挑选 hard-negative（与中心像素尽可能不相似的光谱）。

    说明：
        - 不是直接取 argmin（可能是噪声/异常极值），而是从“最不相似的 q 分位集合”中随机采样；
        - 支持 candidate_mask：只在指定候选集合内采样（例如仅从 *outer ring* 选负样本），
          用于降低“同质背景/近邻同材质”被当作 negative 的概率。

    Args:
        patch: [P,P,B]
        center: [1,B]
        center_hw: center 在 patch 中的坐标 (h,w)
        hard_quantile: 取最不相似的前 q 分位作为候选
        sim_mode: 'corr' 或 'cosine'
        candidate_mask: [P*P] bool，True 表示允许作为 negative 的候选

    Returns:
        neg: [1,B]
    """
    p_h, p_w, b = patch.shape
    patch_unfold = np.reshape(patch, [-1, b], order="F")  # [N,B]
    _m = str(sim_mode).lower()
    if _m in ["cos", "cosine", "raw"]:
        sim = cosine_similarity_rows(patch_unfold, center)  # [N]
    else:
        sim = corr_cosine_similarity_rows(patch_unfold, center)  # [N]

    # 排除中心像素自身
    if center_hw is not None:
        h0, w0 = int(center_hw[0]), int(center_hw[1])
        c_idx = h0 + w0 * p_h  # Fortran-order flatten
        if 0 <= c_idx < sim.shape[0]:
            sim[c_idx] = 1.0

    # (新增) 候选mask：把不允许的位置“抬到最大相似度”，避免被选中
    if candidate_mask is not None:
        cm = np.asarray(candidate_mask).reshape(-1).astype(bool)
        if cm.size != sim.size:
            raise ValueError(f"candidate_mask size mismatch: {cm.size} vs {sim.size}")
        if int(cm.sum()) > 0:
            sim = sim.copy()
            sim[~cm] = 1.0  # 最大相似度（不会落入最小分位集合）

    q = float(np.clip(hard_quantile, 0.0, 1.0))
    if q <= 0.0:
        neg_idx = int(np.argmin(sim))
    else:
        n = sim.shape[0]
        k = int(max(1, round(n * q)))
        hard_ids = np.argpartition(sim, k - 1)[:k]  # 最小 k 个
        neg_idx = int(np.random.choice(hard_ids))

    neg = patch_unfold[neg_idx][None, :]
    return neg.astype(np.float32)


# =========================
# augment ctrl scaling (weak/strong views)
# =========================

def _scale_range_linear(rng, s: float):
    a, b = float(rng[0]), float(rng[1])
    ss = float(s)
    return (a * ss, b * ss)


def _scale_range_around_one(rng, s: float):
    """Scale a (low,high) range around 1.0 (e.g., stripe_strength)."""
    a, b = float(rng[0]), float(rng[1])
    ss = float(s)
    lo = 1.0 + (a - 1.0) * ss
    hi = 1.0 + (b - 1.0) * ss
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def _scale_ctrl(ctrl: Optional[dict], s: float) -> Optional[dict]:
    """缩放 augment.chain_safe 的控制字典，实现 weak/strong 两视角增强。"""
    if ctrl is None:
        return None
    ss = float(max(0.0, s))
    out = dict(ctrl)

    # 概率项：按比例缩放
    for k in ["gaussian_p", "dropout_p", "mult_p", "stripe_p", "smile_p", "keystone_p"]:
        if k in out and out[k] is not None:
            out[k] = float(out[k]) * ss

    # calibration jitter
    if "calib_p" in out and out["calib_p"] is not None:
        out["calib_p"] = float(out["calib_p"]) * ss
        out["calib_p"] = float(np.clip(out["calib_p"], 0.0, 1.0))

    # 连续区间：按比例缩放
    if "gaussian_sigma" in out and isinstance(out["gaussian_sigma"], tuple):
        out["gaussian_sigma"] = (
            max(0.05, float(out["gaussian_sigma"][0]) * ss),
            max(0.05, float(out["gaussian_sigma"][1]) * ss),
        )

    if "mult_std" in out and out["mult_std"] is not None:
        out["mult_std"] = float(out["mult_std"]) * ss

    if "dropout_w" in out and isinstance(out["dropout_w"], tuple):
        w0, w1 = int(out["dropout_w"][0]), int(out["dropout_w"][1])
        w0 = max(1, int(round(w0 * ss)))
        w1 = max(1, int(round(w1 * ss)))
        if w1 < w0:
            w1 = w0
        out["dropout_w"] = (w0, w1)

    if "stripe_strength" in out and isinstance(out["stripe_strength"], tuple):
        out["stripe_strength"] = _scale_range_around_one(out["stripe_strength"], ss)

    # calibration ranges (a,b,c) around 0
    if "calib" in out and isinstance(out["calib"], tuple) and len(out["calib"]) == 3:
        a_range, b_range, c_range = out["calib"]
        out["calib"] = (
            _scale_range_linear(a_range, ss),
            _scale_range_linear(b_range, ss),
            _scale_range_linear(c_range, ss),
        )

    # smile / keystone ranges (around 0, in band-index / pixel units)
    if "smile_shift" in out and isinstance(out["smile_shift"], tuple):
        out["smile_shift"] = _scale_range_linear(out["smile_shift"], ss)

    if "keystone_shift_pix" in out and isinstance(out["keystone_shift_pix"], tuple):
        out["keystone_shift_pix"] = _scale_range_linear(out["keystone_shift_pix"], ss)

    return out


# =========================
# Dataset
# =========================


class Data(data.Dataset):
    """Hyperspectral pixel/patch dataset for self-supervised training."""

    def __init__(
        self,
        path: str,
        w_size: int = 11,
        use_augment: bool = True,
        augment_ctrl: Optional[dict] = None,
        # weak/strong
        augment_view1_scale: float = 0.35,
        augment_view2_scale: float = 1.0,
        # patch encode controls
        patch_encode_tau: float = 1.0,
        patch_encode_sim: str = "corr",
        patch_encode_sigma: Optional[float] = None,
        patch_encode_topk_ratio: Optional[float] = None,
        hard_neg_quantile: float = 0.10,
        # hard-negative 采样区域：
        #   - 'patch'：从整块 patch 里选最不相似的像素（默认，兼容旧版）
        #   - 'ring' ：仅从 outer ring 里选 negative，降低“同质近邻/同材质”被当作 negative 的概率
        hard_neg_region: str = "patch",
        hard_neg_inner_size: Optional[int] = None,
        # return ctx angle for gating
        return_ctx_sim: bool = True,
    ):
        self.w_size = int(w_size)
        self.pad_size = self.w_size // 2
        self.use_augment = bool(use_augment)
        self.augment_ctrl = augment_ctrl
        self.augment_view1_scale = float(augment_view1_scale)
        self.augment_view2_scale = float(augment_view2_scale)

        self.patch_encode_tau = float(patch_encode_tau)
        self.patch_encode_sim = str(patch_encode_sim).lower()
        self.patch_encode_sigma = patch_encode_sigma
        self.patch_encode_topk_ratio = patch_encode_topk_ratio
        self.hard_neg_quantile = float(hard_neg_quantile)
        self.hard_neg_region = str(hard_neg_region).lower()
        self.hard_neg_inner_size = hard_neg_inner_size

        # 预计算 hard-negative 的候选 mask（可选）：只要 w_size 固定，就可以复用
        self._hardneg_candidate_mask = None
        if self.hard_neg_region in ["ring", "outer", "outer_ring"]:
            inner = self.hard_neg_inner_size
            if inner is None:
                # 经验默认：w=11 -> inner=5（中心 5x5 视为“近邻同质区”）
                inner = max(3, int(self.w_size // 2))
            inner = int(inner)
            if inner % 2 == 0:
                inner += 1
            # ensure 1 <= inner < w_size
            if inner >= self.w_size:
                inner = max(1, self.w_size - 2)
                if inner % 2 == 0:
                    inner -= 1
            cy = self.pad_size
            cx = self.pad_size
            ys, xs = np.meshgrid(
                np.arange(self.w_size, dtype=np.int32),
                np.arange(self.w_size, dtype=np.int32),
                indexing="ij",
            )
            ih = inner // 2
            ring = (np.abs(ys - cy) > ih) | (np.abs(xs - cx) > ih)
            self._hardneg_candidate_mask = ring.reshape(-1, order="F").astype(bool)
        self.return_ctx_sim = bool(return_ctx_sim)

        mat = sio.loadmat(path)
        img = mat["data"]  # [H,W,B]
        self.h, self.w, _ = img.shape
        self.nums = self.h * self.w

        # 全局归一化到 [0,1]
        img = standard(img).astype(np.float32)

        # reflect padding
        self.data = np.pad(
            img,
            ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)),
            mode="reflect",
        )
    def set_augment_scales(self, view1_scale: float = None, view2_scale: float = None):
        """动态调整增强强度（curriculum）。

        训练早期用更弱的增强，有助于模型先学到稳定的细粒度谱形；
        训练后期再逐步加大增强强度，提高对噪声/漂移的鲁棒性。
        """
        if view1_scale is not None:
            self.augment_view1_scale = float(view1_scale)
        if view2_scale is not None:
            self.augment_view2_scale = float(view2_scale)


    def __len__(self) -> int:
        return self.nums

    def __getitem__(self, index: int):
        # flat index -> (row,col) with Fortran-order compatible mapping
        position_y = index // self.h
        position_x = index - position_y * self.h
        position_x = position_x + self.pad_size
        position_y = position_y + self.pad_size

        # raw window [w,w,B]
        window_raw = self.data[
            position_x - self.pad_size : position_x + self.pad_size + 1,
            position_y - self.pad_size : position_y + self.pad_size + 1,
            :,
        ]

        center_raw = window_raw[self.pad_size, self.pad_size][None, :].astype(np.float32)  # [1,B]

        # weak / strong windows
        if self.use_augment:
            ctrl1 = _scale_ctrl(self.augment_ctrl, self.augment_view1_scale)
            ctrl2 = _scale_ctrl(self.augment_ctrl, self.augment_view2_scale)
            window_v1 = chain_safe(window_raw, ctrl=ctrl1).astype(np.float32)
            window_v2 = chain_safe(window_raw, ctrl=ctrl2).astype(np.float32)
        else:
            window_v1 = window_raw.astype(np.float32)
            window_v2 = window_raw.astype(np.float32)

        center_v1 = window_v1[self.pad_size, self.pad_size][None, :]  # [1,B]
        center_v2 = window_v2[self.pad_size, self.pad_size][None, :]  # [1,B]

        # -----------------------------------------------------------
        # 重要修正（对齐 HTD‑Mamba 的 SESA 思路）：
        # 1) coded / hard-negative 的选择应基于 **RAW patch 的真实相似结构**。
        #    否则强增强(尤其是 drop/smooth/stripe)会扭曲 patch 内相似关系，
        #    使 SESA 退化，AUC 往往接近 “仅 dropout” 的水平。
        # 2) ctx 门控需要反映 patch 异质性，因此用于计算 ctx_angle 的编码
        #    必须使用 **完整 patch**（不做 top-k 截断），否则角度会被人为压小，
        #    门控形同虚设。
        # -----------------------------------------------------------
        # --- coded view ---
        # 先计算一份“完整 patch”编码（用于 ctx 门控）。
        # 默认 patch_encode_topk_ratio=None，因此这份编码也可以直接作为 coded_v2，避免重复计算。
        coded_gate = None
        if self.return_ctx_sim:
            coded_gate = patch_encoded(
                window_raw,
                center_raw,
                tau=self.patch_encode_tau,
                sim_mode=self.patch_encode_sim,
                spatial_sigma=self.patch_encode_sigma,
                topk_ratio=None,
            ).astype(np.float32)

        if self.patch_encode_topk_ratio is None and coded_gate is not None:
            coded_v2 = coded_gate
        else:
            coded_v2 = patch_encoded(
                window_raw,
                center_raw,
                tau=self.patch_encode_tau,
                sim_mode=self.patch_encode_sim,
                spatial_sigma=self.patch_encode_sigma,
                topk_ratio=self.patch_encode_topk_ratio,
            ).astype(np.float32)

        hardneg_v2 = hard_negative_from_patch(
            window_raw,
            center_raw,
            center_hw=(self.pad_size, self.pad_size),
            hard_quantile=self.hard_neg_quantile,
            sim_mode=self.patch_encode_sim,
            candidate_mask=self._hardneg_candidate_mask,
        ).astype(np.float32)

        if self.return_ctx_sim:
            # 用 RAW 域计算 ctx_angle（更稳定，且不受 top-k 截断“压小角度”的影响）
            # 这里的 coded_gate 已经是 topk_ratio=None 的完整 patch 编码。
            ctx_angle_raw = float(spectral_angle_rows(coded_gate, center_raw)[0])  # radian

            return (
                torch.from_numpy(center_v1).float(),
                torch.from_numpy(center_v2).float(),
                torch.from_numpy(coded_v2).float(),
                torch.from_numpy(hardneg_v2).float(),
                torch.from_numpy(center_raw).float(),
                torch.tensor(ctx_angle_raw, dtype=torch.float32),
            )

        return (
            torch.from_numpy(center_v1).float(),
            torch.from_numpy(center_v2).float(),
            torch.from_numpy(coded_v2).float(),
            torch.from_numpy(hardneg_v2).float(),
            torch.from_numpy(center_raw).float(),
        )