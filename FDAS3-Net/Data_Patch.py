# Data_Patch.py  (train/eval normalization aligned by default; optional fallback;
#                 keep SESA + safe chain + tiny spectral drift)
import numpy as np
import torch
import scipy.io as sio
from typing import Optional
from ts_generation import ts_generation
from augment import chain_safe, spectral_drift_torch  # 新增导入
from Utils import standard  # ★ 新增：与评估端保持一致的标准化

def cosin_similarity(x, y):
    x_norm = np.sqrt(np.sum(x**2, axis=1))
    y_norm = np.sqrt(np.sum(y**2, axis=1))
    return np.sum(x * y, axis=1) / (x_norm * y_norm + 1e-8)

def _make_gaussian(w_size: int, sigma: float) -> np.ndarray:
    c = (w_size - 1) / 2.0
    xs, ys = np.meshgrid(np.arange(w_size), np.arange(w_size), indexing="xy")
    d2 = (xs - c) ** 2 + (ys - c) ** 2
    g = np.exp(-d2 / (2.0 * (sigma**2)))
    g = g / (g.max() + 1e-8)
    return g.astype(np.float32)

def _estimate_noise_prior(data: np.ndarray, gt: np.ndarray, sample_bg: int = 12000) -> float:
    H, W, B = data.shape
    bg = data[gt == 0]
    if bg.size == 0:
        bg = data.reshape(-1, B)
    if bg.shape[0] > sample_bg:
        rng = np.random.RandomState(1)
        bg = bg[rng.choice(bg.shape[0], sample_bg, replace=False)]
    df = np.abs(np.diff(bg, axis=-1)).mean(axis=1)  # [N]
    p95 = np.percentile(df, 95.0) + 1e-6
    score = float(np.clip(np.mean(df) / p95, 0.0, 1.0))
    return score

def _build_aug_ctrl(noise_level: float, cfg: dict) -> dict:
    s = float(cfg.get("aug_calib_strength", 0.7))
    decay = 1.0 - s * float(np.clip(noise_level, 0.0, 1.0))
    base = {
        "gaussian_p": 0.25, "gaussian_sigma": (0.5, 1.2),
        "dropout_p": 0.15, "dropout_max_windows": 1, "dropout_w": (3, 5),
        "mult_p": 0.20, "mult_std": 0.02,
        "calib": ((-0.02, 0.02), (-0.03, 0.03), (-0.01, 0.01)),
    }
    ctrl = dict(base)
    ctrl["gaussian_p"] = base["gaussian_p"] * decay
    lo, hi = base["gaussian_sigma"]
    ctrl["gaussian_sigma"] = (lo * (0.7 + 0.3 * decay), hi * (0.7 + 0.3 * decay))
    ctrl["dropout_p"] = base["dropout_p"] * decay
    ctrl["mult_p"] = base["mult_p"] * decay
    ctrl["mult_std"] = base["mult_std"] * (0.6 + 0.4 * decay)
    return ctrl

def patch_encoded(patch, center, tau: float = 0.5, gauss_w: Optional[np.ndarray] = None, topk: Optional[int] = None):
    if len(patch.shape) == 3 and patch.shape[0] == 1:
        unfold = np.reshape(patch, [-1, patch.shape[2]], order="F")
        spatial_w = np.ones(unfold.shape[0], dtype=np.float32) if gauss_w is None else gauss_w.reshape(-1, order="F")
    else:
        h, w, b = patch.shape
        unfold = np.reshape(patch, [-1, b], order="F")
        spatial_w = np.ones(h * w, dtype=np.float32) if gauss_w is None else gauss_w.reshape(-1, order="F")
    if center.shape[0] == 1 and center.shape[1] != unfold.shape[1]:
        center = np.reshape(center, [-1, center.shape[2]])
    sim = cosin_similarity(unfold, center.reshape(1, -1))
    w_spec = np.exp(sim / max(1e-8, float(tau)))
    w = w_spec * spatial_w
    if topk is not None and topk < w.size:
        idx = np.argpartition(-w, topk)[:topk]
        mask = np.zeros_like(w); mask[idx] = 1.0; w = w * mask + 1e-6
    center_idx = w.size // 2
    w[center_idx] = max(w[center_idx], w.max())
    w = w / (np.sum(w) + 1e-8)
    return np.sum(w[:, None] * unfold, axis=0)[None, :]

class Data:
    """
    Two-view dataset:
      - view0: SESA-encoded patch
      - view1: center pixel with safe spectral augmentations (+ optional tiny spectral drift)
    """
    def __init__(self, path: str, w_size: int, is_train: bool = True, config: dict = None):
        self.path = path
        self.w_size = w_size
        self.is_train = is_train
        self.config = dict(config or {})
        self.band = self.config.get("band", 205)
        mat = sio.loadmat(path)
        self.data_raw = mat["data"]
        self.labels = mat["map"]

        # ★ 训练/评估归一化口径对齐（默认与评估端一致）
        self.train_use_eval_standard = bool(self.config.get("train_use_eval_standard", True))
        if self.train_use_eval_standard:
            self.data = standard(self.data_raw).astype(np.float32)  # 与 select_best/eval 完全对齐
        else:
            vmax = np.percentile(self.data_raw, 99.5)
            self.data = np.clip(self.data_raw / (vmax + 1e-8), 0, 1).astype(np.float32)

        self.h, self.w, _ = self.data.shape
        self.pad = w_size // 2
        self.data = np.pad(self.data, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="symmetric")
        self.n = self.h * self.w

        # 目标谱：单型或小集成
        ts_cfg = self.config.get("ts_ensemble", None)
        try:
            if isinstance(ts_cfg, (list, tuple)) and len(ts_cfg) > 0:
                ts_list = [
                    ts_generation(self.data[self.pad : -self.pad, self.pad : -self.pad, :], self.labels, type=int(t)).astype(np.float32)
                    for t in ts_cfg
                ]
                ts = np.mean(np.stack(ts_list, axis=0), axis=0)
            else:
                ts = ts_generation(self.data[self.pad : -self.pad, self.pad : -self.pad, :], self.labels, type=self.config.get("ts_type", 7)).astype(np.float32)
            self.target = torch.from_numpy(ts.T[None, :, :]).float()  # [1,1,B]
        except Exception:
            self.target = torch.zeros(1, 1, self.band, dtype=torch.float32)

        # SESA
        self.sesa_tau = float(self.config.get("sesa_tau", 0.5))
        self.sesa_topk = int(self.config.get("sesa_topk", 0))
        sigma = float(self.config.get("sesa_sigma", 0.0))
        self.gauss = _make_gaussian(self.w_size, sigma) if sigma > 0 else None

        # 自适应增强
        self.use_aug = bool(self.config.get("use_dynamic_aug", False))
        self.aug_auto = bool(self.config.get("aug_auto_calibrate", False))
        self.noise_prior = _estimate_noise_prior(self.data[self.pad : -self.pad, self.pad : -self.pad, :], self.labels)
        self.aug_ctrl = _build_aug_ctrl(self.noise_prior, self.config) if self.aug_auto else None

        # 线性索引顺序
        # 兼容旧配置项：很多脚本/旧版代码使用 reshape_order 来表示 flatten 的顺序。
        # 若用户只设置了 reshape_order，这里也会同步采用它，避免训练/评估索引错位。
        self.index_order = str(self.config.get("index_order", self.config.get("reshape_order", "F"))).upper()
        # 兼容旧代码：有些地方可能会访问 dataset.reshape_order
        self.reshape_order = self.index_order

        # （新增）谱漂移增强配置（默认关闭）
        self.use_spectral_drift = bool(self.config.get("use_spectral_drift", False))
        self.drift_p = float(self.config.get("drift_p", 0.15))
        self.drift_scale = tuple(self.config.get("drift_scale_range", (-0.02, 0.02)))
        self.drift_shift = tuple(self.config.get("drift_shift_range", (-3.0, 3.0)))
        self.drift_piecewise_p = float(self.config.get("drift_piecewise_p", 0.4))
        self.drift_sinusoid_p = float(self.config.get("drift_sinusoid_p", 0.3))
        self.drift_seg_mag = float(self.config.get("drift_seg_mag", 0.02))

    def __len__(self):
        return self.n

    def _rc_from_idx(self, idx: int):
        if self.index_order == "C":
            row = idx // self.w; col = idx % self.w
        else:
            col = idx // self.h; row = idx - col * self.h
        return row, col

    def __getitem__(self, idx: int):
        row, col = self._rc_from_idx(idx)
        x = row + self.pad; y = col + self.pad
        patch = self.data[x - self.pad : x + self.pad + 1, y - self.pad : y + self.pad + 1, :]
        center = self.data[x, y, :][None, :]

        # view0: 动态 SESA
        if self.use_aug and self.is_train:
            tau = float(np.clip(np.random.normal(self.sesa_tau, 0.05), 0.25, 0.8))
            topk = max(9, self.sesa_topk)
            enc = patch_encoded(patch, center, tau=tau, gauss_w=self.gauss, topk=topk)
        else:
            enc = patch_encoded(patch, center, tau=self.sesa_tau, gauss_w=self.gauss, topk=self.sesa_topk)

        # view1: 安全增强链 (+ 可选小幅谱漂移)
        view1 = center.copy()
        if self.use_aug and self.is_train:
            view1 = chain_safe(view1, ctrl=self.aug_ctrl if self.aug_auto else None)
        view1_t = torch.from_numpy(view1).float()  # [1,B]
        if self.use_aug and self.is_train and self.use_spectral_drift and (np.random.rand() < self.drift_p):
            # torch 实现（可导）；幅度非常小，稳定性优先
            # 重要：spectral_drift_torch 期望输入为 [B,C,L]。
            # 之前写成 [1,1,1,B] 会在函数内部 squeeze 掉维度，导致输出变成 [1,1,B]，
            # 再经过两次 squeeze 会变成 1D [B]，从而触发 DataLoader stack 形状不一致。
            v = view1_t.unsqueeze(0)  # [1,1,B]
            v = spectral_drift_torch(
                v, p=1.0, scale_range=self.drift_scale, shift_range=self.drift_shift,
                piecewise_p=self.drift_piecewise_p, sinusoid_p=self.drift_sinusoid_p, seg_mag=self.drift_seg_mag
            )
            # 保证最终始终为 [1,B]
            view1_t = v.squeeze(0).clamp(0.0, 1.0).view(1, -1)

        # 保险：无论任何增强分支，都强制输出 [1,B]
        view1_t = view1_t.view(1, -1)

        # 保险：view0 也强制为 [1,B]，避免任何潜在的 shape 漂移
        enc_t = torch.from_numpy(enc).float().view(1, -1)

        label = self.labels[row, col]
        return enc_t, view1_t, torch.tensor(label).float()
