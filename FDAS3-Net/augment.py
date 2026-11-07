# augment.py  (rev: adaptive-safe-chain + differentiable spectral drift)
import numpy as np
from typing import Tuple, Optional, Dict
import torch
import torch.nn.functional as F


def _prep_lambda(bands: int):
    lam = np.linspace(0.0, 1.0, bands, dtype=np.float32)
    return lam


def calibration_jitter(
    x: np.ndarray, a_range=(-0.02, 0.02), b_range=(-0.03, 0.03), c_range=(-0.01, 0.01)
):
    """Sensor/illumination calibration jitter: x' = x * (1 + a + b*lam + c*lam^2)."""
    b = x.shape[-1]
    lam = _prep_lambda(b)
    a = np.random.uniform(*a_range)
    bb = np.random.uniform(*b_range)
    c = np.random.uniform(*c_range)
    scale = (1.0 + a + bb * lam + c * (lam**2)).astype(np.float32)
    return np.clip(x * scale, 0.0, 1.0)


def spectral_dropout(x: np.ndarray, p=0.25, max_windows=2, w_min=3, w_max=7):
    """随机置零窄频段并用线性插值回填（结构化遮挡的温和模拟）。"""
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


def gaussian_smooth(x: np.ndarray, p=0.3, sigma_range=(0.5, 1.2)):
    """轻度谱向平滑，模拟分辨率差异。"""
    if np.random.rand() > p:
        return x
    from scipy.ndimage import gaussian_filter1d

    s = np.random.uniform(*sigma_range)
    return np.clip(gaussian_filter1d(x, sigma=s, axis=-1, mode="nearest"), 0.0, 1.0)


def mult_noise(x: np.ndarray, p=0.4, std=0.01):
    if np.random.rand() > p:
        return x
    n = np.random.lognormal(mean=0.0, sigma=std, size=x.shape).astype(np.float32)
    return np.clip(x * n, 0.0, 1.0)


def chain_safe(x: np.ndarray, ctrl: Optional[Dict] = None) -> np.ndarray:
    """
    安全增强链：标定微扰 + 轻平滑 + 窄带遮挡 + 乘性噪声。
    若提供 ctrl，可自适应调幅：
      ctrl = {
        'calib': (a_range,b_range,c_range),
        'gaussian_p': float, 'gaussian_sigma': (low, high) or float,
        'dropout_p': float, 'dropout_max_windows': int, 'dropout_w': (wmin,wmax),
        'mult_p': float, 'mult_std': float
      }
    """
    a_range = (-0.02, 0.02)
    b_range = (-0.03, 0.03)
    c_range = (-0.01, 0.01)
    g_p, g_sig = 0.25, (0.5, 1.2)
    d_p, d_win, d_w = 0.15, 1, (3, 5)
    m_p, m_std = 0.20, 0.02

    if ctrl:
        if "calib" in ctrl and isinstance(ctrl["calib"], tuple):
            a_range, b_range, c_range = ctrl["calib"]
        g_p = float(ctrl.get("gaussian_p", g_p))
        g_sig = ctrl.get("gaussian_sigma", g_sig)
        d_p = float(ctrl.get("dropout_p", d_p))
        d_win = int(ctrl.get("dropout_max_windows", d_win))
        d_w = ctrl.get("dropout_w", d_w)
        m_p = float(ctrl.get("mult_p", m_p))
        m_std = float(ctrl.get("mult_std", m_std))

    x = calibration_jitter(x, a_range=a_range, b_range=b_range, c_range=c_range)
    if isinstance(g_sig, tuple):
        x = gaussian_smooth(x, p=g_p, sigma_range=g_sig)
    else:
        # 固定 sigma
        from scipy.ndimage import gaussian_filter1d

        if np.random.rand() < g_p:
            x = np.clip(
                gaussian_filter1d(x, sigma=float(g_sig), axis=-1, mode="nearest"),
                0.0,
                1.0,
            )
    x = spectral_dropout(
        x, p=d_p, max_windows=d_win, w_min=int(d_w[0]), w_max=int(d_w[1])
    )
    x = mult_noise(x, p=m_p, std=m_std)
    return x


# ----------------------- 可微分的谱漂移 / 分段形变 ----------------------- #
def _resample_1d_torch(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    线性插值重采样（可导）
    x: [B, C, L], idx: [B, L]，取样坐标（0..L-1）
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
    谱漂移/形变：仿射 (缩放+平移) +（可选）分段/正弦形变；可导。
    x: [B,1,L] 或 [B,C,L]
    """
    if p <= 0.0 or torch.rand(1).item() > p:
        return x
    if x.dim() == 3:
        B, C, L = x.shape
    elif x.dim() == 4 and x.size(2) == 1:  # [B,1,1,L] 罕见
        x = x.squeeze(2)
        B, C, L = x.shape
    else:
        B, C, L = x.shape

    device = x.device
    base = torch.linspace(0, L - 1, L, device=device).view(1, L)
    center = (L - 1) / 2.0

    scale = (
        torch.rand(B, device=device) * (scale_range[1] - scale_range[0])
        + scale_range[0]
    )
    shift = (
        torch.rand(B, device=device) * (shift_range[1] - shift_range[0])
        + shift_range[0]
    )

    idx = (base - center) * (1.0 + scale.unsqueeze(1)) + center + shift.unsqueeze(1)

    # 额外形变：分段线性 or 正弦扰动
    if torch.rand(1).item() < piecewise_p:
        # 正弦扰动（平滑、稳定）
        if torch.rand(1).item() < sinusoid_p:
            k = torch.randint(low=1, high=4, size=(B,), device=device)  # 1~3 次谐波
            amp = (torch.rand(B, device=device) * 2 - 1) * (seg_mag * L)
            phase = torch.rand(B, device=device) * 2 * np.pi
            sin_term = amp.unsqueeze(1) * torch.sin(
                2 * np.pi * k.unsqueeze(1) * base / L + phase.unsqueeze(1)
            )
            idx = idx + sin_term
        else:
            # 分段线性：三段控制点
            p1 = (torch.rand(B, device=device) * 0.4 + 0.2) * (L - 1)
            p2 = (torch.rand(B, device=device) * 0.4 + 0.4) * (L - 1)
            o1 = (torch.rand(B, device=device) * 2 - 1) * (seg_mag * L)
            o2 = (torch.rand(B, device=device) * 2 - 1) * (seg_mag * L)
            idx_add = torch.zeros_like(idx)
            for b in range(B):
                t = base[0]
                # 0..p1 线性
                m1 = o1[b] / (p1[b] + 1e-6)
                # p1..p2
                m2 = (o2[b] - o1[b]) / (p2[b] - p1[b] + 1e-6)
                # p2..L-1 回到 0
                m3 = -o2[b] / (L - 1 - p2[b] + 1e-6)
                idx_add[b] = torch.where(
                    t <= p1[b],
                    m1 * t,
                    torch.where(
                        t <= p2[b], o1[b] + m2 * (t - p1[b]), o2[b] + m3 * (t - p2[b])
                    ),
                )
            idx = idx + idx_add

    idx = idx.clamp(0, L - 1)
    return _resample_1d_torch(x, idx)
