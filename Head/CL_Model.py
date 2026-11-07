# -*- coding: utf-8 -*-
"""CL_Model.py

HTD-Mamba 改良版 backbone encoder（用于高光谱像素级目标检测/匹配）。

核心设计：
1) 多谱表示统一建模：raw + (可选) wavelet 低/高频 + (可选) 一/二阶谱导
2) 双向 Pyramid-SSM（Mamba）序列建模：减弱单向建模偏置，增强细粒度谱形表达
3) (可选) 背景原型 bank：配合训练期的 BPSA（SwAV 风格交换聚类）学习异质背景结构

注：训练阶段不需要目标光谱；目标光谱只在测试/评估阶段用于 score 计算。
"""

from __future__ import annotations

from typing import Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DPyramidSSM import Mamba
from augment import DiffSWT1D, MultiLevelSWT, spectral_drift_torch


# =========================
# small helpers
# =========================

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + eps)



def _savgol_coeffs(window: int, poly: int, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    """Compute Savitzky–Golay FIR coefficients (least-squares polynomial fit).

    We implement a tiny subset of SciPy's savgol_coeffs so we can keep the model
    self-contained (no SciPy dependency in forward).

    Args:
        window: odd window length
        poly: polynomial order (< window)
        deriv: derivative order (0 -> smoothing)
        delta: sample spacing (1 for band index)

    Returns:
        coeffs: [window] float32
    """
    window = int(window)
    poly = int(poly)
    deriv = int(deriv)
    if window % 2 == 0 or window < 3:
        raise ValueError(f"window must be odd and >=3, got {window}")
    if poly >= window:
        raise ValueError(f"poly must be < window, got poly={poly}, window={window}")
    if deriv < 0 or deriv > poly:
        raise ValueError(f"deriv must be in [0, poly], got deriv={deriv}, poly={poly}")

    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    # Vandermonde with increasing powers: [window, poly+1]
    A = np.vander(x, N=poly + 1, increasing=True).astype(np.float32)
    pinv = np.linalg.pinv(A).astype(np.float32)  # [poly+1, window]
    coeff = pinv[deriv] * (math.factorial(deriv) / (delta ** deriv))
    return coeff.astype(np.float32)


def _gaussian_kernel1d(window: int, sigma: float) -> torch.Tensor:
    window = int(window)
    if window % 2 == 0:
        window += 1
    sigma = float(max(1e-6, sigma))
    half = window // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / (k.sum() + 1e-8)
    return k


class FixedDepthwiseConv1d(nn.Module):
    """Fixed depthwise 1D convolution with reflect padding."""

    def __init__(self, channels: int, kernel_1d: torch.Tensor, padding_mode: str = "reflect"):
        super().__init__()
        self.channels = int(channels)
        k = kernel_1d.flatten().to(dtype=torch.float32)
        if k.numel() % 2 == 0:
            raise ValueError("kernel length must be odd")
        self.pad = int(k.numel() // 2)
        w = k.view(1, 1, -1).repeat(self.channels, 1, 1)
        self.register_buffer("weight", w)
        self.padding_mode = str(padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L], depthwise conv per channel
        x = F.pad(x, (self.pad, self.pad), mode=self.padding_mode)
        return F.conv1d(x, self.weight, groups=self.channels)



class SpectralGroupAttention(nn.Module):
    """HTD-Mamba backbone + embedding + background prototypes.

    输入: x [B,1,C] 或 [B,C]
    输出: z [B,D]

    结构:
      Wavelet/Deriv -> Group tokenization (Conv1d) -> (Bi-)Mamba -> MLP proj

    额外:
      bg_prototypes: [K,D] 可学习背景原型（用于 BPSA 损失与推理背景抑制）
    """

    def __init__(
        self,
        band: int,
        group_length: int,
        channel_dim: int,
        state_size: int,
        device: torch.device,
        layer: int = 1,
        # input normalization
        use_input_norm: bool = True,
        # wavelet
        use_wavelet: bool = True,
        wavelet_level: int = 1,
        wavelet_type: str = "db4",
        wavelet_learnable: bool = False,
        # derivatives
        use_deriv: bool = True,
        deriv_scale: float = 1.0,
        # NEW: smooth-before-derivative
        deriv_smooth: str = "savgol",
        deriv_smooth_window: int = 7,
        deriv_smooth_poly: int = 2,
        deriv_gaussian_sigma: float = 1.0,
        # channel gate
        use_channel_gate: bool = True,
        # model-side spectral drift aug
        drift_p: float = 0.25,
        # drift 参数：尽量“弱扰动 + 可控”，避免破坏窄谱保真（smile/keystone/微漂移可用小幅模拟）
        drift_shift_range: Tuple[float, float] = (-1.0, 1.0),
        drift_scale_range: Tuple[float, float] = (-0.01, 0.01),
        drift_piecewise_p: float = 0.35,
        drift_sinusoid_p: float = 0.25,
        drift_seg_mag: float = 0.01,
        # bidirectional
        bidirectional: bool = True,
        # group tokenization
        token_stride: Optional[int] = None,
        token_activation: str = "leaky_relu",
        # embedding
        embed_dim: int = 64,
        # background prototypes
        proto_bank_size: int = 64,
        # compatibility
        **kwargs,
    ):
        super().__init__()

        self.band = int(band)
        self.group_length = int(group_length)
        self.channel_dim = int(channel_dim)
        self.state_size = int(state_size)
        self.device = device
        self.use_input_norm = bool(use_input_norm)  # 是否对输入光谱做“去均值+L2”归一化（与 correlation-cosine 一致）
        self.layer = int(layer)

        self.use_wavelet = bool(use_wavelet)
        self.wavelet_level = int(max(1, wavelet_level))
        self.wavelet_type = str(wavelet_type)

        self.use_deriv = bool(use_deriv)
        self.deriv_scale = float(deriv_scale)

        self.use_channel_gate = bool(use_channel_gate)
        self.drift_p = float(drift_p)
        # drift 超参数（建议配合 curriculum：早期学谱形，后期再增强鲁棒性）
        self.drift_shift_range = (float(drift_shift_range[0]), float(drift_shift_range[1]))
        self.drift_scale_range = (float(drift_scale_range[0]), float(drift_scale_range[1]))
        self.drift_piecewise_p = float(drift_piecewise_p)
        self.drift_sinusoid_p = float(drift_sinusoid_p)
        self.drift_seg_mag = float(drift_seg_mag)
        self.bidirectional = bool(bidirectional)

        # group-wise tokenization hyper-params
        # 原版 HTD-Mamba 的 stride 通常取 group_length//4（重叠分组，更细粒度谱形）
        # 这里允许 token_stride < group_length 做重叠 token（更细粒度，但更难训）
        if token_stride is None:
            # 与原始 HTD-Mamba 代码一致：stride = group_length//4（重叠分组，提升细粒度谱形感知）
            token_stride = max(1, self.group_length // 4)
        self.token_stride = int(max(1, int(token_stride)))
        self.token_activation = str(token_activation).lower()

        # -------- wavelet front-end --------
        if self.use_wavelet:
            # Multi-level SWT: levels 1..wavelet_level (stacked on channel dimension)
            self.wavelet = MultiLevelSWT(
                wavelet=wavelet_type,
                channels=1,
                max_level=self.wavelet_level,
                learnable=bool(wavelet_learnable),
            )
        else:
            self.wavelet = None

        # -------- derivatives (smooth -> deriv) --------
        self.deriv_smooth_mode = str(deriv_smooth).lower()
        self.deriv_smooth_window = int(deriv_smooth_window)
        self.deriv_smooth_poly = int(deriv_smooth_poly)
        self.deriv_gaussian_sigma = float(deriv_gaussian_sigma)

        if self.use_deriv and self.deriv_smooth_mode != "none":
            if self.deriv_smooth_mode == "savgol":
                k_np = _savgol_coeffs(
                    window=self.deriv_smooth_window,
                    poly=self.deriv_smooth_poly,
                    deriv=0,
                    delta=1.0,
                )
                k = torch.from_numpy(k_np)
            elif self.deriv_smooth_mode == "gaussian":
                k = _gaussian_kernel1d(self.deriv_smooth_window, self.deriv_gaussian_sigma)
            else:
                k = None
            self.deriv_smoother = FixedDepthwiseConv1d(1, k) if k is not None else None
        else:
            self.deriv_smoother = None

        # fixed derivative kernels (central difference)
        self.register_buffer(
            "d1_weight",
            torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32).view(1, 1, 3),
        )
        self.register_buffer(
            "d2_weight",
            torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32).view(1, 1, 3),
        )

        # -------- group tokenization --------
        wave_ch = (2 * self.wavelet_level) if self.use_wavelet else 0
        in_ch = 1 + wave_ch + (2 if self.use_deriv else 0)

        if self.use_channel_gate:
            # SE-style channel gate: 根据数据自适应加权 low/high/deriv 通道
            se_hidden = max(4, in_ch * 4)
            self.chan_gate = nn.Sequential(
                nn.Linear(in_ch, se_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(se_hidden, in_ch),
                nn.Sigmoid(),
            )
        else:
            self.chan_gate = None

        stride = self.token_stride
        self.stride = int(stride)
        self.group_division = nn.Conv1d(
            in_channels=in_ch,
            out_channels=self.channel_dim,
            kernel_size=self.group_length,
            stride=stride,
            padding=0,
            bias=True,
        )

        # 与论文一致：Conv1d 后接一个轻量非线性（默认 LeakyReLU）
        if self.token_activation in ["leaky_relu", "lrelu"]:
            self.token_act = nn.LeakyReLU(0.1, inplace=True)
        elif self.token_activation in ["relu"]:
            self.token_act = nn.ReLU(inplace=True)
        elif self.token_activation in ["gelu"]:
            self.token_act = nn.GELU()
        elif self.token_activation in ["none", "identity", "linear"]:
            self.token_act = nn.Identity()
        else:
            raise ValueError(f"Unknown token_activation: {token_activation}")

        # token 序列长度
        self.seq_len = (self.band - self.group_length) // stride + 1
        if self.seq_len <= 0:
            raise ValueError(
                f"Invalid seq_len={self.seq_len}. Check band={band}, group_length={group_length}, stride={stride}."
            )

        # -------- (Bi-)Mamba encoder --------
        self.mamba_fwd = Mamba(self.seq_len, self.channel_dim, self.state_size, layer=self.layer)
        if self.bidirectional:
            self.mamba_bwd = Mamba(self.seq_len, self.channel_dim, self.state_size, layer=self.layer)
            self.bi_fuse = nn.Linear(2 * self.channel_dim, self.channel_dim)

            # NEW: adaptive bi-direction fusion gate (per-sample & per-channel)
            # fixes the incompleteness of a global scalar gate: different spectra /
            # different noise regimes may prefer forward/backward evidence differently.
            hidden = max(8, self.channel_dim)
            self.bi_gate_net = nn.Sequential(
                nn.Linear(2 * self.channel_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, self.channel_dim),
                nn.Sigmoid(),
            )
        else:
            self.mamba_bwd = None
            self.bi_fuse = None
            self.bi_gate_net = None

        # -------- projection head --------
        self.embed_dim = int(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.seq_len * self.channel_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.embed_dim),
        )

        # -------- learnable background prototypes (embedding space) --------
        self.proto_bank_size = int(max(1, proto_bank_size))
        self.bg_prototypes = nn.Parameter(torch.randn(self.proto_bank_size, self.embed_dim) * 0.02)

    # ----------------------
    # prototype utilities
    # ----------------------
    def get_bg_prototypes(self, normalize: bool = True) -> torch.Tensor:
        """返回背景原型 [K,D]。推理时建议 normalize=True。"""
        p = self.bg_prototypes
        if normalize:
            p = F.normalize(p, dim=1)
        return p

    # ----------------------
    # encoding
    # ----------------------
    def _encode(self, x: torch.Tensor, force_no_drift: bool = False) -> torch.Tensor:
        """x -> embedding z.

        Args:
            x: [B,C] or [B,1,C]
            force_no_drift: True 时禁止 model-side spectral drift（评估/可视化更稳定）
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,C]
        if x.dim() != 3 or x.shape[2] != self.band:
            raise ValueError(f"x must be [B,1,C] with C==band, got {tuple(x.shape)}")


        # 0) model-side spectral drift (train only)
        # 注意：漂移应作用在“原始光谱”上，并在其后重新做 input_norm，
        # 否则会造成 train/eval 分布不一致（尤其是 drift curriculum 后期更明显）。
        if self.training and (not force_no_drift) and self.drift_p > 0:
            if torch.rand(()) < self.drift_p:
                x = spectral_drift_torch(
                    x,
                    p=1.0,
                    scale_range=self.drift_scale_range,
                    shift_range=self.drift_shift_range,
                    piecewise_p=self.drift_piecewise_p,
                    sinusoid_p=self.drift_sinusoid_p,
                    seg_mag=self.drift_seg_mag,
                )

        # 1) input normalization：去均值 + L2（使模型更关注谱形而非亮度/偏置）
        if self.use_input_norm:
            x = x - x.mean(dim=2, keepdim=True)
            x = F.normalize(x, dim=2, eps=1e-6)

        # 2) build input feature channels: raw + multi-scale wavelet + derivatives
        feats = [x]  # keep raw spectrum for narrow-band fidelity

        # multi-scale SWT (levels 1..wavelet_level), returns stacked channels
        if self.use_wavelet and self.wavelet is not None:
            low, high = self.wavelet(x)  # [B, L, C] (L = levels)
            # keep wavelet channels scale-invariant (important for narrow-band cues)
            if self.use_input_norm:
                low = low - low.mean(dim=2, keepdim=True)
                high = high - high.mean(dim=2, keepdim=True)
                low = F.normalize(low, dim=2, eps=1e-6)
                high = F.normalize(high, dim=2, eps=1e-6)
            feats.append(low)
            feats.append(high)

        # 3) fine-grained derivatives (d1, d2): smooth -> deriv (noise-aware)
        if self.use_deriv:
            xs = x
            if self.deriv_smoother is not None:
                xs = self.deriv_smoother(xs)
                # keep scale-invariant shape cues after smoothing
                if self.use_input_norm:
                    xs = xs - xs.mean(dim=2, keepdim=True)
                    xs = F.normalize(xs, dim=2, eps=1e-6)

            d1 = F.conv1d(F.pad(xs, (1, 1), mode="reflect"), self.d1_weight)
            d2 = F.conv1d(F.pad(xs, (1, 1), mode="reflect"), self.d2_weight)
            if self.deriv_scale != 1.0:
                d1 = d1 * self.deriv_scale
                d2 = d2 * self.deriv_scale
            feats.append(d1)
            feats.append(d2)

        x_feat = torch.cat(feats, dim=1)

        # 4) channel gate
        if self.chan_gate is not None:
            g = self.chan_gate(x_feat.mean(dim=2))  # [B,in_ch]
            x_feat = x_feat * g.unsqueeze(-1)

        # 5) group tokenization: [B,channel_dim,seq_len]
        feat = self.group_division(x_feat)  # [B,D,L]
        feat = self.token_act(feat)
        feat = feat.permute(0, 2, 1)        # [B,L,D]

        # 6) (bi-)mamba
        if self.bidirectional:
            fwd = self.mamba_fwd(feat)
            bwd = torch.flip(self.mamba_bwd(torch.flip(feat, dims=[1])), dims=[1])

            # adaptive per-channel gate from global pooled context
            pool = torch.cat([fwd.mean(dim=1), bwd.mean(dim=1)], dim=-1)  # [B,2D]
            gate = self.bi_gate_net(pool).unsqueeze(1)  # [B,1,D]

            fused = torch.cat([gate * fwd, (1.0 - gate) * bwd], dim=-1)
            feat = self.bi_fuse(fused)
        else:
            feat = self.mamba_fwd(feat)

        # 7) projection
        z = self.proj(feat.reshape(feat.shape[0], -1))
        return z

    def forward(self, x: torch.Tensor, force_no_drift: bool = False) -> torch.Tensor:
        return self._encode(x, force_no_drift=force_no_drift)