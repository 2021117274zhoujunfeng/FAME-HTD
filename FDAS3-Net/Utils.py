# Utils.py (rev4): Robust post-processing / visualization utilities
# ---------------------------------------------------------------
# 说明：
# - AUC 评估使用 det_map（已标准化到 [0,1]），不要在这里改动 eval 的 det_map 计算口径
# - 本文件只负责“可视化用”的稳健拉伸/抑制（DSC + NBS++ / Rank 等）
# - 针对你日志里出现的情况：
#   * abu-beach-4（你称为 Pavia）: det_map 极度稀疏，raw_p99 接近 0，导致原先的 NBS/分位数拉伸把背景“抬亮”，可视化变得一塌糊涂
#   * 因此在 auto 模式下增加“稀疏图”识别与 rank-gamma 回退，使背景更暗、目标更亮
#
# 支持的 cfg 参数（可在 *.py 的 cfg 里调参）：
#   postproc_mode: 'auto' | 'dsc_nbs' | 'rank'
#   post_low_p / post_hi_p: 分位数拉伸范围（默认 1~99）
#   rank_gamma: rank 模式下的 gamma（>1 会让背景更暗、突出 top-k）
#   use_dsc, dsc_bg_top_p, dsc_T, dsc_gamma, dsc_clamp_p
#   nbs_delta, median_k, nbs_clamp_min
#   sparse_p99_thr, sparse_vis_mean_thr: 识别“极稀疏 det_map”并回退到 rank 的阈值
#   post_auto_zero_thr: auto 模式下，若 near-min 占比过大则回退到 rank
# ---------------------------------------------------------------

from __future__ import annotations
import numpy as np
import os
from typing import Optional, Dict

# ---------- Basic utilities ----------
def standard(X: np.ndarray) -> np.ndarray:
    """Plain min-max to [0,1] with safety for flat arrays."""
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).any():
        return np.zeros_like(X, dtype=np.float32)
    mn = float(np.nanmin(X))
    mx = float(np.nanmax(X))
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx <= mn):
        return np.zeros_like(X, dtype=np.float32)
    return ((X - mn) / (mx - mn)).astype(np.float32)

def checkFile(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def cosin_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape[1] == y.shape[1]
    x_norm = np.linalg.norm(x, axis=1) + 1e-8
    y_norm = np.linalg.norm(y, axis=1) + 1e-8
    xy = np.sum(x * y, axis=1)
    return (xy / (x_norm * y_norm)).astype(np.float32)

# ---------- Percentile helpers ----------
def _percentile(x: np.ndarray, p: float) -> float:
    p = float(np.clip(p, 0, 100))
    return float(np.percentile(np.asarray(x).reshape(-1), p))

def robust_minmax(x: np.ndarray, low_p: float = 1.0, hi_p: float = 99.0) -> np.ndarray:
    """
    Percentile-based clipping + min-max; robust for heavy tails and negative regimes.
    """
    x = np.asarray(x, dtype=np.float32)
    lo = _percentile(x, low_p)
    hi = _percentile(x, hi_p)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        return standard(x)
    y = np.clip(x, lo, hi)
    return ((y - lo) / (hi - lo)).astype(np.float32)

# ---------- DSC (robust for negative distributions) ----------
def dsc_calibration(
    y_score: np.ndarray,
    bg_top_p: float = 98.0,
    T: float = 1.2,
    gamma: float = 2.0,
    clamp_p: float = 99.5,
) -> np.ndarray:
    """
    Robust DSC:
      1) winsorize extreme top tail to clamp_p for stability
      2) compute background ceiling b = P(bg_top_p)
      3) piecewise transform on [lo, b] using a robust low anchor lo=P5
      4) preserve residual above b
    Works even if b < 0.
    """
    y = np.asarray(y_score, dtype=np.float32).reshape(-1)
    y = np.nan_to_num(
        y,
        nan=0.0,
        posinf=float(np.nanmax(y[np.isfinite(y)])) if np.isfinite(y).any() else 0.0,
        neginf=0.0,
    )

    # winsorize the very top tail
    top = _percentile(y, clamp_p)
    if np.isfinite(top):
        y = np.minimum(y, top)

    lo = _percentile(y, 5.0)
    b  = _percentile(y, bg_top_p)
    T = max(1e-4, float(T))
    gamma = max(1.0, float(gamma))

    y0 = np.minimum(y, b)          # background piece
    y1 = y - y0                    # residual head (>=0 if y<=top)

    denom = float(b - lo)
    # b 和 lo 太接近时，DSC 没意义，直接返回
    if (not np.isfinite(denom)) or (abs(denom) < 1e-6):
        return y.astype(np.float32)

    z = np.clip((y0 - lo) / denom, 0.0, 1.0)
    z = z ** (1.0 / T)
    z = z ** (1.0 / gamma)
    y0_out = z * denom + lo

    return (y0_out + y1).astype(np.float32)

# ---------- NBS++ ----------
def nonlinear_bg_suppression(
    y_score: np.ndarray,
    delta: float = 0.05,
    median_k: int = 3,
    bg_ref_p: float = 98.0,
    clamp_min: bool = False,
) -> np.ndarray:
    """
    Subtract a small fraction of the background ceiling to push background down.
    默认不做 clamp（保持对“负分数”场景的兼容），但对极稀疏 det_map 的可视化，
    建议在 cfg 中把 nbs_clamp_min=True（否则背景可能被抬亮到 0.5 左右）。
    """
    y = np.asarray(y_score, dtype=np.float32)
    b = _percentile(y, bg_ref_p)
    y2 = y - float(delta) * b

    if clamp_min:
        # 对可视化更友好：把负值截断到 0（不会影响 AUC，因为 AUC 不用这里）
        y2 = np.maximum(y2, 0.0)

    # optional tiny median filter to suppress salt-and-pepper artifacts
    if median_k is not None and int(median_k) > 1:
        k = int(median_k)

        def _median_filter2d(x: np.ndarray, k: int) -> np.ndarray:
            if x.ndim == 1:
                pad = k // 2
                xp = np.pad(x, pad, mode="reflect")
                out = np.empty_like(x)
                for i in range(x.shape[0]):
                    out[i] = np.median(xp[i:i+k])
                return out
            elif x.ndim == 2:
                H, W = x.shape
                pad = k // 2
                xp = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")
                out = np.empty_like(x)
                for i in range(H):
                    for j in range(W):
                        out[i, j] = np.median(xp[i:i+k, j:j+k])
                return out
            else:
                shp = x.shape
                return _median_filter2d(x.reshape(-1), k).reshape(shp)

        try:
            y2 = _median_filter2d(y2, k)
        except Exception:
            pass

    return y2.astype(np.float32)

# ---------- Rank-based fallback ----------
def rank_normalize(y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Map scores to [0,1] by rank (monotonic), robust to pathologic distributions.
    gamma>1：让绝大多数背景更暗，仅突出 top-k（推荐 2~6）

    注意：Rank 映射是“相对排序”可视化，可能会把“占比很大的弱响应背景”映射成较亮区域。
    对于“det_map 极稀疏/背景整体很低”的目标检测场景，更推荐使用 raw/pow/clip_pow。
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    order = np.argsort(y)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(y.size, dtype=np.float32)
    out = ranks / max(1, y.size - 1)
    g = float(gamma)
    if np.isfinite(g) and g > 1.0:
        out = out ** g
    return out.astype(np.float32)

def pow_normalize(y: np.ndarray, gamma: float = 3.0) -> np.ndarray:
    """绝对值域的幂次压缩：y -> y^gamma（gamma>1 抑制背景，突出高响应）"""
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, 0.0, 1.0)
    g = float(gamma)
    if np.isfinite(g) and g != 1.0:
        y = y ** g
    return y.astype(np.float32)

def clip_pow_normalize(y: np.ndarray, clip_p: float = 99.9, gamma: float = 4.0, eps: float = 1e-8) -> np.ndarray:
    """
    Top-Percentile 剪切 + 幂次压缩（非常适合“稀疏检测图”的可视化）：
      1) 取阈值 t = percentile(y, clip_p)
      2) 只把 [t,1] 拉伸到 [0,1]，t 以下全部压到 0
      3) 再做 y^gamma，让背景更暗

    这样不会像 rank 那样把“弱响应但占比大”的背景抬亮。
    """
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, 0.0, 1.0)
    t = float(_percentile(y, float(clip_p)))
    denom = max(float(eps), 1.0 - t)
    y = (y - t) / denom
    y = np.clip(y, 0.0, 1.0)
    g = float(gamma)
    if np.isfinite(g) and g != 1.0:
        y = y ** g
    return y.astype(np.float32)

def _apply_simple_viz_mode(y: np.ndarray, cfg: dict, mode: str) -> np.ndarray:
    """对 raw/pow/clip_pow/rank 这类“轻量可视化模式”统一入口。"""
    m = (mode or "raw").lower()
    if m == "raw":
        # 可选：对 raw 也做一次 pow（默认不做）
        g = float(cfg.get("raw_gamma", 1.0))
        return pow_normalize(y, gamma=g)
    if m == "pow":
        return pow_normalize(y, gamma=float(cfg.get("pow_gamma", 3.0)))
    if m == "clip_pow":
        return clip_pow_normalize(
            y,
            clip_p=float(cfg.get("clip_p", 99.9)),
            gamma=float(cfg.get("pow_gamma", 4.0)),
        )
    if m == "rank":
        return rank_normalize(y, gamma=float(cfg.get("rank_gamma", 3.0))).reshape(y.shape)
    # default fallback
    return np.clip(np.asarray(y, dtype=np.float32), 0.0, 1.0)

# ---------- End-to-end pipeline ----------
def dsc_then_nbs(
    y_score: np.ndarray,
    *,
    cfg: Optional[dict] = None,
    mode: Optional[str] = None,
    robust_low_p: Optional[float] = None,
    robust_hi_p: Optional[float] = None,
) -> np.ndarray:
    """
    Visualization pipeline (仅影响可视化，不影响 AUC):
      mode ∈ {'dsc_nbs','rank','raw','pow','clip_pow','auto'}
      - 'raw'     : 直接使用 det_map（可选 raw_gamma 做轻微压缩）
      - 'pow'     : det_map^pow_gamma（绝对值域压缩，抑制背景）
      - 'clip_pow': 先按 clip_p 做高分位剪切再压缩（最适合稀疏检测图）
      - 'rank'    : rank-normalize（相对排序，可视化可能抬亮大面积弱响应）
      - 'dsc_nbs' : DSC -> NBS++ -> robust_minmax([low_p,hi_p])
      - 'auto'    : 先走 dsc_nbs；若检测到“稀疏图/背景抬亮/塌缩”等异常，则回退到更稳健的模式
    """
    c = cfg or {}
    mode = (mode or c.get("postproc_mode") or "auto").lower()

    # ---- 轻量模式（推荐用于你现在的 abu-beach-4 可视化问题）----
    if mode in ("raw", "pow", "clip_pow", "rank"):
        y = np.asarray(y_score, dtype=np.float32)
        return _apply_simple_viz_mode(y, c, mode).reshape(y.shape)

    low_p = float(robust_low_p if robust_low_p is not None else c.get("post_low_p", 1.0))
    hi_p  = float(robust_hi_p  if robust_hi_p  is not None else c.get("post_hi_p", 99.0))

    y = np.asarray(y_score, dtype=np.float32)

    # ---- DSC + NBS ----
    y1 = y
    if c.get("use_dsc", True):
        y1 = dsc_calibration(
            y1,
            bg_top_p=c.get("dsc_bg_top_p", 98.0),
            T=c.get("dsc_T", 1.2),
            gamma=c.get("dsc_gamma", 2.0),
            clamp_p=c.get("dsc_clamp_p", 99.5),
        )

    y2 = nonlinear_bg_suppression(
        y1,
        delta=c.get("nbs_delta", 0.05),
        median_k=c.get("median_k", 3),
        bg_ref_p=c.get("dsc_bg_top_p", 98.0),
        clamp_min=bool(c.get("nbs_clamp_min", False)),
    )

    y_vis = robust_minmax(y2, low_p, hi_p)

    if mode != "auto":
        return y_vis.reshape(y.shape)

    # ---- auto fallback heuristics ----
    # 1) 塌缩检测：可视化后，太多像元贴近最小值
    near_min = float((y_vis <= (y_vis.min() + 1e-6)).sum() / max(1, y_vis.size))
    thr = float(c.get("post_auto_zero_thr", 0.85))
    # 2) 动态范围塌缩：p95-p5 太小
    p5 = _percentile(y2, 5.0)
    p95 = _percentile(y2, 95.0)
    collapse_by_range = (p95 - p5) < 1e-6

    # 3) “极稀疏图”识别：raw_p99 极小，但 y_vis 均值却很高（典型背景被抬亮）
    raw_p99 = _percentile(y, 99.0)
    vis_mean = float(np.mean(y_vis))

    sparse_p99_thr = float(c.get("sparse_p99_thr", 1e-5))
    sparse_vis_mean_thr = float(c.get("sparse_vis_mean_thr", 0.12))
    sparse_case = (raw_p99 < sparse_p99_thr) and (vis_mean > sparse_vis_mean_thr)

    # 4) 额外：背景抬亮（近零很少但均值很大，且 raw_p99 不大）
    lifted_bg = (vis_mean > 0.25) and (near_min < 0.01) and (raw_p99 < 0.02)

    # 默认回退策略：
    # - 稀疏/抬亮：优先 clip_pow（最贴合“目标检测图”的视觉预期）
    # - 塌缩：用 rank 让你至少能看出相对结构
    if sparse_case or lifted_bg:
        fb = str(c.get("post_sparse_fallback", "clip_pow")).lower()
        return _apply_simple_viz_mode(y, c, fb).reshape(y.shape)

    if collapse_by_range or (near_min > thr):
        fb = str(c.get("post_collapse_fallback", "rank")).lower()
        return _apply_simple_viz_mode(y, c, fb).reshape(y.shape)

    return y_vis.reshape(y.shape)

# ---------- Diagnostics for eval logging ----------
def summarize_scores(y_raw: np.ndarray, y_vis: np.ndarray) -> Dict[str, float]:
    y_raw = np.asarray(y_raw, dtype=np.float32).reshape(-1)
    y_vis = np.asarray(y_vis, dtype=np.float32).reshape(-1)

    def _p(a, v): 
        return float(np.percentile(a, v))

    near_min = float((y_vis <= (y_vis.min() + 1e-6)).sum() / max(1, y_vis.size))
    return {
        "raw_min": float(np.min(y_raw)),
        "raw_max": float(np.max(y_raw)),
        "raw_mean": float(np.mean(y_raw)),
        "raw_p95": _p(y_raw, 95.0),
        "raw_p99": _p(y_raw, 99.0),
        "vis_min": float(np.min(y_vis)),
        "vis_max": float(np.max(y_vis)),
        "vis_mean": float(np.mean(y_vis)),
        "vis_near_min_ratio": near_min,
    }
