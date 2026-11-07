# augment.py
# 模块功能：包含全面的数据变换（安全增强链、谱漂移）和小波增强模块
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# PyWavelets 用于 SWT/小波谱证据。
# 有些环境未必预装 pywt：为了保证 RAW / RAW+GRAD 也能跑，这里做成可选依赖。
try:
    import pywt  # type: ignore
except Exception:
    pywt = None
from typing import Tuple, Optional, Dict
import os
# ==============================================================================
# Part 1: 数据变换与增强 (Data Transformation & Augmentation)
# ==============================================================================


def _prep_lambda(bands: int):
    """辅助函数，生成用于标定抖动的波长相关系数。"""
    lam = np.linspace(0.0, 1.0, bands, dtype=np.float32)
    return lam


def calibration_jitter(
    x: np.ndarray,
    a_range=(-0.02, 0.02),
    b_range=(-0.03, 0.03),
    c_range=(-0.01, 0.01),
) -> np.ndarray:
    """
    传感器/光照标定抖动：模拟传感器响应的细微非线性和整体亮度变化。
    公式: x' = x * (1 + a + b*lam + c*lam^2).
    """
    b = x.shape[-1]
    lam = _prep_lambda(b)
    a = np.random.uniform(*a_range)
    bb = np.random.uniform(*b_range)
    c = np.random.uniform(*c_range)
    scale = (1.0 + a + bb * lam + c * (lam**2)).astype(np.float32)
    return np.clip(x * scale, 0.0, 1.0)


def spectral_dropout(
    x: np.ndarray,
    p: float = 0.25,
    max_windows: int = 2,
    w_min: int = 3,
    w_max: int = 7,
) -> np.ndarray:
    """
    光谱结构化遮挡：随机置零窄频段并用线性插值回填，模拟数据中的小范围缺失或噪声。
    """
    if np.random.rand() > p:
        return x
    x = x.copy()
    B = x.shape[-1]
    for _ in range(np.random.randint(1, max_windows + 1)):
        w = np.random.randint(w_min, w_max + 1)
        s = np.random.randint(0, max(1, B - w))
        e = s + w
        # 边界线性插值
        left = x[..., s - 1 : s] if s > 0 else x[..., s : s + 1]
        right = x[..., e : e + 1] if e < B else x[..., e - 1 : e]
        ramp = np.linspace(0, 1, w, dtype=np.float32)
        fill = left * (1 - ramp) + right * ramp
        x[..., s:e] = fill
    return np.clip(x, 0.0, 1.0)


def gaussian_smooth(
    x: np.ndarray,
    p: float = 0.3,
    sigma_range: Tuple[float, float] = (0.5, 1.2),
) -> np.ndarray:
    """
    高斯平滑：对光谱进行轻度平滑，模拟不同仪器间的光谱分辨率差异。
    """
    if np.random.rand() > p:
        return x
    from scipy.ndimage import gaussian_filter1d

    s = np.random.uniform(*sigma_range)
    return np.clip(gaussian_filter1d(x, sigma=s, axis=-1, mode="nearest"), 0.0, 1.0)


def mult_noise(x: np.ndarray, p: float = 0.4, std: float = 0.01) -> np.ndarray:
    """
    乘性噪声：模拟光子噪声等与信号强度相关的噪声。
    """
    if np.random.rand() > p:
        return x
    n = np.random.lognormal(mean=0.0, sigma=std, size=x.shape).astype(np.float32)
    return np.clip(x * n, 0.0, 1.0)


def stripe_noise(
    x: np.ndarray,
    p: float = 0.15,
    max_stripes: int = 2,
    strength_range: Tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """
    条纹噪声增强：在空间维度上随机选取若干行或列整体缩放，
    用于模拟推扫式成像传感器中常见的条带增益不均匀。

    仅当输入维度 >= 3（例如 [H,W,B] patch）且随机触发时生效。
    """
    if np.random.rand() > p or x.ndim < 3:
        return x

    x = x.copy()
    # 随机选择行方向或列方向
    axis = -3 if np.random.rand() < 0.5 else -2
    size = x.shape[axis]
    num = np.random.randint(1, max_stripes + 1)
    idxs = np.random.choice(size, size=num, replace=False)

    for idx in idxs:
        sl = [slice(None)] * x.ndim
        sl[axis] = idx
        scale = np.random.uniform(*strength_range)
        x[tuple(sl)] = np.clip(x[tuple(sl)] * scale, 0.0, 1.0)

    return x



# ------------------------------------------------------------------------------
# NEW: Physical-consistent augmentations for smile / keystone (patch-level)
# ------------------------------------------------------------------------------
# smile:   spatially varying spectral shift Δλ = f(x,y)
# keystone: band-dependent spatial shift   Δx = g(λ)
#
# These distortions are common in pushbroom hyperspectral sensors and can break
# narrow-band fidelity, causing poor cross-process robustness if not modeled.
# We implement lightweight, vectorized numpy versions so they can run inside
# DataLoader augmentation without heavy Python loops.
# ------------------------------------------------------------------------------

def _linear_resample_last_axis(x2d: np.ndarray, idx2d: np.ndarray) -> np.ndarray:
    """Vectorized linear resample along last axis.

    Args:
        x2d:   [N,B]
        idx2d: [N,B] float indices in [0,B-1]

    Returns:
        y2d:   [N,B]
    """
    if x2d.ndim != 2 or idx2d.ndim != 2:
        raise ValueError(f"x2d/idx2d must be 2D, got {x2d.shape} / {idx2d.shape}")
    N, B = x2d.shape
    if idx2d.shape != (N, B):
        raise ValueError(f"idx2d shape mismatch: {idx2d.shape} vs {(N,B)}")

    idx0 = np.floor(idx2d).astype(np.int64)
    idx1 = idx0 + 1
    w = (idx2d - idx0).astype(np.float32)

    idx0 = np.clip(idx0, 0, B - 1)
    idx1 = np.clip(idx1, 0, B - 1)

    v0 = np.take_along_axis(x2d, idx0, axis=1)
    v1 = np.take_along_axis(x2d, idx1, axis=1)
    return v0 * (1.0 - w) + v1 * w


def smile_distortion(
    x: np.ndarray,
    p: float = 0.0,
    shift_range: Tuple[float, float] = (-0.8, 0.8),
) -> np.ndarray:
    """Simulate *smile* distortion on a patch.

    Smile: center wavelength shifts vary across the spatial field of view, i.e.,
           Δλ = f(x,y). Locally, this manifests as a spatially varying spectral shift.

    Implementation:
        - sample a smooth spatial field (linear combination of x/y ramps),
        - normalize to [-1,1],
        - scale by a random amplitude in `shift_range` (units: *band index*),
        - per-pixel resample spectrum with linear interpolation.

    Notes:
        - Only applies to patch inputs with x.ndim>=3 (e.g., [H,W,B]).
        - For 1D spectra, this op is skipped (no spatial context).
    """
    if p <= 0.0 or np.random.rand() > p or x.ndim < 3:
        return x

    # Assume last dim is bands; take the last 3 dims as [H,W,B]
    patch = x.copy().astype(np.float32)
    H, W, B = patch.shape[-3], patch.shape[-2], patch.shape[-1]

    if H <= 1 or W <= 1 or B <= 3:
        return patch

    # Smooth spatial field
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, H, dtype=np.float32),
        np.linspace(-1.0, 1.0, W, dtype=np.float32),
        indexing="ij",
    )
    a, b = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
    field = a * xx + b * yy
    field = field / (np.max(np.abs(field)) + 1e-6)

    amp = float(np.random.uniform(shift_range[0], shift_range[1]))
    shift_map = (amp * field).astype(np.float32)  # [H,W] in band-index units

    flat = patch.reshape(-1, B)
    shift_flat = shift_map.reshape(-1, 1)
    base = np.arange(B, dtype=np.float32)[None, :]

    # Positive shift should move features to *higher* indices:
    # t'(k) = t(k - shift)
    idx = base - shift_flat
    idx = np.clip(idx, 0.0, float(B - 1))

    out = _linear_resample_last_axis(flat, idx).reshape(H, W, B)
    return np.clip(out, 0.0, 1.0)


def _shift_patch_per_band(
    patch: np.ndarray,
    shift_curve: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    """Shift a [H,W,B] patch spatially with band-dependent offsets.

    Args:
        patch: [H,W,B]
        shift_curve: [B] shifts in *pixel units* (can be fractional)
        axis: 0 (rows/y) or 1 (cols/x)

    Returns:
        shifted patch [H,W,B]
    """
    patch = patch.astype(np.float32, copy=False)
    H, W, B = patch.shape
    curve = shift_curve.astype(np.float32).reshape(B, 1, 1)

    if axis == 1:
        # shift along W (x / columns)
        cube = patch.transpose(2, 0, 1)  # [B,H,W]
        grid = np.arange(W, dtype=np.float32).reshape(1, 1, W)
        idx = grid - curve  # sample coordinate in original image
        idx = np.clip(idx, 0.0, float(W - 1))

        idx0 = np.floor(idx).astype(np.int64)
        idx1 = np.clip(idx0 + 1, 0, W - 1)
        w = (idx - idx0).astype(np.float32)

        idx0 = np.broadcast_to(idx0, cube.shape)
        idx1 = np.broadcast_to(idx1, cube.shape)

        v0 = np.take_along_axis(cube, idx0, axis=2)
        v1 = np.take_along_axis(cube, idx1, axis=2)
        out = v0 * (1.0 - w) + v1 * w
        return out.transpose(1, 2, 0)  # [H,W,B]

    if axis == 0:
        # shift along H (y / rows)
        cube = patch.transpose(2, 1, 0)  # [B,W,H]
        grid = np.arange(H, dtype=np.float32).reshape(1, 1, H)
        idx = grid - curve
        idx = np.clip(idx, 0.0, float(H - 1))

        idx0 = np.floor(idx).astype(np.int64)
        idx1 = np.clip(idx0 + 1, 0, H - 1)
        w = (idx - idx0).astype(np.float32)

        idx0 = np.broadcast_to(idx0, cube.shape)
        idx1 = np.broadcast_to(idx1, cube.shape)

        v0 = np.take_along_axis(cube, idx0, axis=2)
        v1 = np.take_along_axis(cube, idx1, axis=2)
        out = v0 * (1.0 - w) + v1 * w
        return out.transpose(2, 1, 0)  # [H,W,B]

    return patch


def keystone_distortion(
    x: np.ndarray,
    p: float = 0.0,
    shift_pix_range: Tuple[float, float] = (-0.8, 0.8),
    axis: int = 1,
    mode: str = "linear_sin",
    freq_range: Tuple[float, float] = (0.5, 1.5),
) -> np.ndarray:
    """Simulate *keystone* distortion on a patch.

    Keystone: spatial misregistration changes with wavelength, i.e.,
              Δx = g(λ). This creates band-to-band spatial shifts.

    Implementation:
        - sample a smooth band-dependent shift curve (linear + sinusoid),
        - apply band-dependent spatial shifts via vectorized linear interpolation.

    Notes:
        - Only applies to patch inputs with x.ndim>=3 (e.g., [H,W,B]).
        - axis=1 -> x/columns shift, axis=0 -> y/rows shift.
    """
    if p <= 0.0 or np.random.rand() > p or x.ndim < 3:
        return x

    patch = x.copy().astype(np.float32)
    H, W, B = patch.shape[-3], patch.shape[-2], patch.shape[-1]
    if H <= 2 or W <= 2 or B <= 3:
        return patch

    # Normalize wavelength coordinate in [-1,1]
    t = np.linspace(-1.0, 1.0, B, dtype=np.float32)

    amp = float(np.random.uniform(shift_pix_range[0], shift_pix_range[1]))
    curve = np.zeros((B,), dtype=np.float32)

    m = str(mode).lower()
    if "linear" in m:
        curve += amp * t

    if "sin" in m:
        f0, f1 = float(freq_range[0]), float(freq_range[1])
        freq = float(np.random.uniform(min(f0, f1), max(f0, f1)))
        phase = float(np.random.uniform(0.0, 2.0 * np.pi))
        curve += 0.5 * amp * np.sin(2.0 * np.pi * freq * (t + 1.0) * 0.5 + phase).astype(np.float32)

    lo = float(min(shift_pix_range[0], shift_pix_range[1]))
    hi = float(max(shift_pix_range[0], shift_pix_range[1]))
    curve = np.clip(curve, lo, hi).astype(np.float32)

    # Apply band-dependent spatial shifts
    axis_i = int(axis)
    return np.clip(_shift_patch_per_band(patch, curve, axis=axis_i), 0.0, 1.0)



def chain_safe(x: np.ndarray, ctrl: Optional[Dict] = None) -> np.ndarray:
    """
    安全增强链：将多种增强方法串联起来，可根据控制字典自适应调整强度。
    默认包含：
      - calibration_jitter: 细尺度标定变化
      - gaussian_smooth:    光谱分辨率差异
      - spectral_dropout:   局部波段缺失 / 强噪声
      - mult_noise:         乘性随机噪声
      - stripe_noise:       空间条纹噪声（推扫条带）
    """
    a_range = (-0.02, 0.02)
    b_range = (-0.03, 0.03)
    c_range = (-0.01, 0.01)
    # NEW: probabilistic calibration jitter (so the weak view can preserve
    # fine-grained spectral shapes when desired)
    calib_p = 1.0
    g_p, g_sig = 0.25, (0.5, 1.2)
    d_p, d_win, d_w = 0.15, 1, (3, 5)
    m_p, m_std = 0.20, 0.02
    s_p, s_max, s_str = 0.15, 2, (0.85, 1.15)
    # NEW: smile / keystone (physical-consistent) augment controls
    smile_p, smile_shift = 0.0, (-0.8, 0.8)
    keystone_p, keystone_shift_pix = 0.0, (-0.8, 0.8)
    keystone_axis = 1  # 1: x/cols, 0: y/rows
    keystone_mode = "linear_sin"
    keystone_freq = (0.5, 1.5)

    if ctrl:
        if "calib" in ctrl and isinstance(ctrl["calib"], tuple):
            a_range, b_range, c_range = ctrl["calib"]
        if "calib_p" in ctrl and ctrl["calib_p"] is not None:
            calib_p = float(ctrl.get("calib_p", calib_p))
        g_p = float(ctrl.get("gaussian_p", g_p))
        g_sig = ctrl.get("gaussian_sigma", g_sig)
        d_p = float(ctrl.get("dropout_p", d_p))
        d_win = int(ctrl.get("dropout_max_windows", d_win))
        d_w = ctrl.get("dropout_w", d_w)
        m_p = float(ctrl.get("mult_p", m_p))
        m_std = float(ctrl.get("mult_std", m_std))
        s_p = float(ctrl.get("stripe_p", s_p))
        s_max = int(ctrl.get("stripe_max_stripes", s_max))
        s_str = ctrl.get("stripe_strength", s_str)
        # smile / keystone
        smile_p = float(ctrl.get("smile_p", smile_p))
        smile_shift = ctrl.get("smile_shift", smile_shift)
        keystone_p = float(ctrl.get("keystone_p", keystone_p))
        keystone_shift_pix = ctrl.get("keystone_shift_pix", keystone_shift_pix)
        keystone_axis = ctrl.get("keystone_axis", keystone_axis)
        keystone_mode = ctrl.get("keystone_mode", keystone_mode)
        keystone_freq = ctrl.get("keystone_freq_range", keystone_freq)

    # 标定抖动 (NEW: optionally probabilistic)
    calib_p = float(np.clip(calib_p, 0.0, 1.0))
    if calib_p > 0.0 and (calib_p >= 1.0 or np.random.rand() < calib_p):
        x = calibration_jitter(x, a_range=a_range, b_range=b_range, c_range=c_range)

    # 频域平滑
    if isinstance(g_sig, tuple):
        x = gaussian_smooth(x, p=g_p, sigma_range=g_sig)
    else:
        from scipy.ndimage import gaussian_filter1d

        if np.random.rand() < g_p:
            x = np.clip(
                gaussian_filter1d(x, sigma=float(g_sig), axis=-1, mode="nearest"),
                0.0,
                1.0,
            )

    # 局部波段缺失 + 乘性噪声 + 条纹噪声
    x = spectral_dropout(
        x,
        p=d_p,
        max_windows=d_win,
        w_min=int(d_w[0]),
        w_max=int(d_w[1]),
    )
    x = mult_noise(x, p=m_p, std=m_std)
    x = stripe_noise(x, p=s_p, max_stripes=s_max, strength_range=s_str)

    # NEW: physical-consistent smile / keystone (only affects patch inputs)
    x = smile_distortion(x, p=smile_p, shift_range=smile_shift)
    ax = keystone_axis
    if isinstance(ax, str):
        ax_s = ax.lower()
        if ax_s in ["x", "col", "cols", "w", "width"]:
            ax = 1
        elif ax_s in ["y", "row", "rows", "h", "height"]:
            ax = 0
    try:
        ax_i = int(ax)
    except Exception:
        ax_i = 1
    x = keystone_distortion(
        x,
        p=keystone_p,
        shift_pix_range=keystone_shift_pix,
        axis=ax_i,
        mode=str(keystone_mode),
        freq_range=keystone_freq,
    )

    return x


# --- 可微分的谱漂移 / 分段形变 (Differentiable Spectral Drift) ---


def _resample_1d_torch(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    可导的1D线性插值重采样。
    x: [B, C, L], idx: [B, L]，浮点型采样坐标 (范围 0..L-1)。
    """
    B, C, L = x.shape
    idx0 = torch.floor(idx).clamp(0, L - 1)
    idx1 = (idx0 + 1).clamp(0, L - 1)
    w = (idx - idx0).unsqueeze(1)  # [B,1,L]
    idx0 = idx0.long().unsqueeze(1).expand(-1, C, -1)
    idx1 = idx1.long().unsqueeze(1).expand(-1, C, -1)
    x0 = torch.gather(x, 2, idx0)
    x1 = torch.gather(x, 2, idx1)
    return x0 * (1.0 - w) + x1 * w


def spectral_drift_torch(
    x: torch.Tensor,
    p: float = 1.0,
    scale_range: Tuple[float, float] = (-0.02, 0.02),
    shift_range: Tuple[float, float] = (-3.0, 3.0),
    piecewise_p: float = 0.5,
    sinusoid_p: float = 0.35,
    seg_mag: float = 0.02,
) -> torch.Tensor:
    """
    可导的光谱漂移/形变：通过重采样实现，包含全局仿射(缩放+平移)和局部分段/正弦形变。
    """
    if p <= 0.0 or torch.rand(1).item() > p:
        return x

    if x.dim() == 4 and x.size(2) == 1:
        x = x.squeeze(2)
    B, C, L = x.shape

    device = x.device
    base = torch.linspace(0, L - 1, L, device=device).view(1, L)
    center = (L - 1) / 2.0

    # 全局仿射
    scale = (
        torch.rand(B, device=device) * (scale_range[1] - scale_range[0])
        + scale_range[0]
    )
    shift = (
        torch.rand(B, device=device) * (shift_range[1] - shift_range[0])
        + shift_range[0]
    )
    idx = (base - center) * (1.0 + scale.unsqueeze(1)) + center + shift.unsqueeze(1)

    # 额外形变
    if torch.rand(1).item() < piecewise_p:
        # 正弦扰动
        if torch.rand(1).item() < sinusoid_p:
            k = torch.randint(low=1, high=4, size=(B,), device=device)
            amp = (torch.rand(B, device=device) * 2 - 1) * (seg_mag * L)
            phase = torch.rand(B, device=device) * 2 * np.pi
            sin_term = amp.unsqueeze(1) * torch.sin(
                2 * np.pi * k.unsqueeze(1) * base / L + phase.unsqueeze(1)
            )
            idx = idx + sin_term
        # 分段线性
        else:
            p1 = (torch.rand(B, device=device) * 0.4 + 0.2) * (L - 1)
            p2 = (torch.rand(B, device=device) * 0.4 + 0.4) * (L - 1)
            o1 = (torch.rand(B, device=device) * 2 - 1) * (seg_mag * L)
            o2 = (torch.rand(B, device=device) * 2 - 1) * (seg_mag * L)
            idx_add = torch.zeros_like(idx)
            for b in range(B):
                t = base[0]
                m1 = o1[b] / (p1[b] + 1e-6)
                m2 = (o2[b] - o1[b]) / (p2[b] - p1[b] + 1e-6)
                m3 = -o2[b] / (L - 1 - p2[b] + 1e-6)
                idx_add[b] = torch.where(
                    t <= p1[b],
                    m1 * t,
                    torch.where(
                        t <= p2[b],
                        o1[b] + m2 * (t - p1[b]),
                        o2[b] + m3 * (t - p2[b]),
                    ),
                )
            idx = idx + idx_add

    idx = idx.clamp(0, L - 1)
    return _resample_1d_torch(x, idx)


# ==============================================================================
# Part 2: 小波增强模块 (Wavelet Enhancement Module)
# ==============================================================================


def _upsample_filter_1d(h: torch.Tensor, level: int) -> torch.Tensor:
    """对小波滤波器进行上采样，以实现多级SWT。"""
    if level <= 1:
        return h
    z = torch.zeros(
        h.numel() + (h.numel() - 1) * (2 ** (level - 1) - 1),
        dtype=h.dtype,
        device=h.device,
    )
    z[:: 2 ** (level - 1)] = h
    return z


class DiffSWT1D(nn.Module):
    """
    可微分的一维静态小波变换 (Stationary Wavelet Transform, SWT) 层。
    使用卷积操作实现，滤波器权重可以设置为可学习的。
    """

    def __init__(
        self,
        channels: int,
        wavelet: str = "db4",
        level: int = 1,
        learnable: bool = True,
        lo_scale: float = 1.0,
        hi_scale: float = 1.0,
    ):
        super().__init__()
        w = pywt.Wavelet(wavelet)
        lp = torch.tensor(list(reversed(w.dec_lo)), dtype=torch.float32)
        hp = torch.tensor(list(reversed(w.dec_hi)), dtype=torch.float32)
        lp = _upsample_filter_1d(lp, level)
        hp = _upsample_filter_1d(hp, level)

        k = int(lp.numel())
        # NOTE:
        # For an odd kernel, symmetric padding (k-1)//2 is "same".
        # For an even kernel (e.g. db4 has 8 taps), symmetric padding produces
        # length mismatch and an implicit phase shift.
        #
        # We implement an explicit "same" padding by padding (k-1) samples in total
        # (left/right can be asymmetric when k is even), then running conv with
        # padding=0. This keeps the SWT alignment more stable for narrow-band cues.
        pad_total = int(max(0, k - 1))
        self.left_pad = pad_total // 2
        self.right_pad = pad_total - self.left_pad

        self.low = nn.Conv1d(
            channels, channels, kernel_size=k, padding=0, groups=channels, bias=False
        )
        self.high = nn.Conv1d(
            channels, channels, kernel_size=k, padding=0, groups=channels, bias=False
        )

        with torch.no_grad():
            self.low.weight.copy_(lp.view(1, 1, k).repeat(channels, 1, 1))
            self.high.weight.copy_(hp.view(1, 1, k).repeat(channels, 1, 1))

        for p in self.parameters():
            p.requires_grad = bool(learnable)

        self.lo_scale = lo_scale
        self.hi_scale = hi_scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回低频分量和高频分量。
        """
        if self.left_pad > 0 or self.right_pad > 0:
            # reflect padding is usually the best choice for spectral sequences;
            # fallback to replicate for very short sequences.
            try:
                x_pad = F.pad(x, (self.left_pad, self.right_pad), mode="reflect")
            except Exception:
                x_pad = F.pad(x, (self.left_pad, self.right_pad), mode="replicate")
        else:
            x_pad = x

        low = self.low(x_pad)
        high = self.high(x_pad)

        # Safety: keep length consistent (should already match)
        if low.size(-1) != x.size(-1):
            low = low[..., : x.size(-1)]
            high = high[..., : x.size(-1)]

        low = low * self.lo_scale
        high = high * self.hi_scale

        return low, high

class MultiLevelSWT(nn.Module):
    """Multi-level Stationary Wavelet Transform wrapper.

    This module stacks SWT coefficients from level=1..max_level.
    For each level it returns a low-pass and high-pass output with the same
    spectral length as the input (SWT). Outputs are concatenated on the channel
    dimension.

    Input:
        x: [B, channels, L]

    Output:
        low_stack : [B, channels*max_level, L]
        high_stack: [B, channels*max_level, L]
    """

    def __init__(
        self,
        channels: int,
        wavelet: str = "db4",
        max_level: int = 1,
        learnable: bool = False,
        lo_scale: float = 1.0,
        hi_scale: float = 1.0,
    ):
        super().__init__()
        # 如果没有安装 pywt，则无法使用小波谱证据。
        # 但 RAW / RAW+GRAD 消融仍然可以跑。
        if pywt is None:
            raise ImportError(
                "PyWavelets(pywt) is not installed. Install with `pip install PyWavelets` to enable wavelet features."
            )
        self.channels = int(channels)
        self.max_level = int(max(1, max_level))
        self.levels = nn.ModuleList(
            [
                DiffSWT1D(
                    channels=self.channels,
                    wavelet=wavelet,
                    level=lv,
                    learnable=learnable,
                    lo_scale=lo_scale,
                    hi_scale=hi_scale,
                )
                for lv in range(1, self.max_level + 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lows, highs = [], []
        for layer in self.levels:
            lo, hi = layer(x)
            lows.append(lo)
            highs.append(hi)
        low_stack = torch.cat(lows, dim=1)
        high_stack = torch.cat(highs, dim=1)
        return low_stack, high_stack


def standard(X):
    '''
    Standardization
    universal
    :param X:
    :return:
    '''
    min_x = np.min(X)
    max_x = np.max(X)
    if min_x == max_x:
        return np.zeros_like(X)
    return np.float32((X - min_x) / (max_x - min_x))

def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def cosin_similarity(x, y):
    norm_x = np.sqrt(np.sum(x ** 2, axis=-1))
    norm_y = np.sqrt(np.sum(y ** 2, axis=-1))
    x_y = np.sum(np.multiply(x, y), axis=-1)
    similarity = np.clip(x_y / (norm_x * norm_y), -1, 1)
    return np.arccos(similarity)


def ts_generation(data, gt, type=0):
    '''
    using different methods to select the target spectrum
    :param type:
    0-Averaged spectrum
    1-Averaged spectrum after spatial debiasing (L2 norm)
    2-Averaged spectrum after spectral debiasing (spectral angle)
    3-Spatial median spectrum (median of L2 norm)
    4-Spectral median spectrum (median of spectral angles)
    5-The target pixel with the closest Euclidean distance to all the other target pixels
    6-The target pixel with the closest cosine distance to all the target pixels
    7-The target pixel closest to the averaged spectrum (L2 norm)
    8-The target pixel closest to the averaged spectrum (spectral angle)
    :return:
    '''
    ind = np.where(gt == 1)
    ts = data[ind]
    avg_target_spectrum = np.mean(ts, axis=0)
    avg_target_spectrum = np.expand_dims(avg_target_spectrum, axis=-1)
    if type == 0:
        return avg_target_spectrum
    elif type == 1:
        spatial_distance = np.sqrt(np.sum((ts - avg_target_spectrum.T) ** 2, axis=-1))
        arg_distance = np.argsort(spatial_distance)
        saved_num = int(ts.shape[0] * 0.8)
        saved_spectrums = ts[arg_distance[:saved_num]]
        removed_deviation_target_spectrum = np.mean(saved_spectrums, axis=0)
        removed_deviation_target_spectrum = np.expand_dims(removed_deviation_target_spectrum, axis=-1)
        return removed_deviation_target_spectrum
    elif type == 2:
        spatial_distance = cosin_similarity(ts, avg_target_spectrum.T)
        arg_distance = np.argsort(spatial_distance)
        saved_num = int(ts.shape[0] * 0.8)
        saved_spectrums = ts[arg_distance[:saved_num]]
        removed_deviation_target_spectrum = np.mean(saved_spectrums, axis=0)
        removed_deviation_target_spectrum = np.expand_dims(removed_deviation_target_spectrum, axis=-1)
        return removed_deviation_target_spectrum
    elif type == 3:
        dist_list = np.zeros([ts.shape[0]])
        for i in range(ts.shape[0]):
            dist_list[i] = np.mean(np.sqrt(np.sum(np.square(ts - ts[i]), axis=-1)))
        arg_distance = np.argsort(dist_list)
        mid = ts.shape[0] // 2
        mid_target_spectrum = ts[arg_distance[mid]]
        mid_target_spectrum = np.expand_dims(mid_target_spectrum, axis=-1)
        return mid_target_spectrum
    elif type == 4:
        dist_list = np.zeros([ts.shape[0]])
        for i in range(ts.shape[0]):
            dist_list[i] = np.mean(cosin_similarity(ts, ts[i]))
        arg_distance = np.argsort(dist_list)
        mid = ts.shape[0] // 2
        mid_target_spectrum = ts[arg_distance[mid]]
        mid_target_spectrum = np.expand_dims(mid_target_spectrum, axis=-1)
        return mid_target_spectrum
    elif type == 5:
        min_distance = 10000
        opd_i = 0
        for i in range(ts.shape[0]):
            dist = np.mean(np.sqrt(np.sum(np.square(ts - ts[i]), axis=-1)))
            # print(dist)
            if dist < min_distance:
                min_distance = dist
                opd_i = i
        target_spectrum = ts[opd_i]
        target_spectrum = np.expand_dims(target_spectrum, axis=-1)
        return target_spectrum
    elif type == 6:
        min_distance = 10000
        opd_i = 0
        for i in range(ts.shape[0]):
            dist = np.mean(cosin_similarity(ts, ts[i]))
            # print(dist)
            if dist < min_distance:
                min_distance = dist
                opd_i = i
        target_spectrum = ts[opd_i]
        target_spectrum = np.expand_dims(target_spectrum, axis=-1)
        return target_spectrum

    elif type == 7:
        distance = np.sqrt(np.sum((ts - avg_target_spectrum.T) ** 2, axis=-1))
        arg_distance = np.argsort(distance)
        avg_L2_target_spectrum = ts[arg_distance[0]]
        avg_L2_target_spectrum = np.expand_dims(avg_L2_target_spectrum, axis=-1)
        return avg_L2_target_spectrum
    elif type == 8:
        distance = cosin_similarity(ts, avg_target_spectrum.T)
        # print(distance)
        arg_distance = np.argsort(distance)
        avg_cosin_target_spectrum = ts[arg_distance[0]]
        avg_cosin_target_spectrum = np.expand_dims(avg_cosin_target_spectrum, axis=-1)
        return avg_cosin_target_spectrum
    else:
        return avg_target_spectrum
