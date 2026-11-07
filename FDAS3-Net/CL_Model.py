# CL_Model.py  (fix: gate uses current sample's hi/lo ratio; richer debug unchanged)
import torch
from torch import nn
import torch.nn.functional as F
import pywt
from PyramidSSM import Mamba
import math

def _upsample_filter_1d(h: torch.Tensor, level: int) -> torch.Tensor:
    if level <= 1:
        return h
    z = torch.zeros(
        h.numel() + (h.numel() - 1) * (2 ** (level - 1) - 1),
        dtype=h.dtype,
        device=h.device,
    )
    z[:: 2 ** (level - 1)] = h
    return z

class SpectralAttention(nn.Module):
    """目标感知频谱注意力（TASA）"""
    def __init__(self, band: int, reduction: int = 16, use_target: bool = True):
        super().__init__()
        self.band = band
        self.use_target = use_target
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        hidden = max(8, band // reduction)
        self.fc = nn.Sequential(
            nn.Linear(band, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, band, bias=False),
        )
        if use_target:
            self.fc_t = nn.Linear(band, band, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._last_att = None

    def forward(self, x: torch.Tensor, target_vec: torch.Tensor = None):
        # x: [B,1,Band]
        b, _, d = x.size()
        avg_out = self.avg_pool(x.permute(0, 2, 1)).squeeze(-1)
        max_out = self.max_pool(x.permute(0, 2, 1)).squeeze(-1)
        att = self.fc(avg_out) + self.fc(max_out)
        if self.use_target and (target_vec is not None):
            if target_vec.dim() == 3:
                target_vec = target_vec.squeeze(0).squeeze(0)
            if target_vec.dim() == 1:
                target_vec = target_vec.unsqueeze(0)
            t = target_vec.to(x.device).expand(b, -1)
            t = self.fc_t(t)
            att = att + t
        att = self.sigmoid(att).unsqueeze(1)  # [B,1,Band]
        self._last_att = att.detach().squeeze(1)
        return x * att

    @property
    def last_att(self):
        return self._last_att

class DiffSWT1D(nn.Module):
    def __init__(self, channels: int, wavelet: str = "db4", level: int = 1,
                 learnable: bool = True, lo_scale: float = 1.0, hi_scale: float = 1.0):
        super().__init__()
        w = pywt.Wavelet(wavelet)
        lp = torch.tensor(list(reversed(w.dec_lo)), dtype=torch.float32)
        hp = torch.tensor(list(reversed(w.dec_hi)), dtype=torch.float32)
        lp = _upsample_filter_1d(lp, level); hp = _upsample_filter_1d(hp, level)
        k = int(lp.numel()); pad = (k - 1) // 2
        self.low = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, groups=channels, bias=False)
        self.high = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, groups=channels, bias=False)
        with torch.no_grad():
            self.low.weight.copy_(lp.view(1, 1, k).repeat(channels, 1, 1))
            self.high.weight.copy_(hp.view(1, 1, k).repeat(channels, 1, 1))
        for p in self.parameters(): p.requires_grad = bool(learnable)
        self.lo_scale = lo_scale; self.hi_scale = hi_scale

    def forward(self, x: torch.Tensor):
        low = self.low(x); high = self.high(x)
        if low.size(-1) != x.size(-1):
            if low.size(-1) > x.size(-1):
                trim = (low.size(-1) - x.size(-1)) // 2
                low = low[:, :, trim : trim + x.size(-1)]
                high = high[:, :, trim : trim + x.size(-1)]
            else:
                pad = x.size(-1) - low.size(-1); left = pad // 2; right = pad - left
                low = F.pad(low, (left, right)); high = F.pad(high, (left, right))
        low = low * self.lo_scale; high = high * self.hi_scale
        return low, high

class SE1d(nn.Module):
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        hidden = max(8, channels // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w, w

class BackgroundSuppressor(nn.Module):
    def __init__(self, channel_dim: int, band: int):
        super().__init__()
        self.bg_encoder = nn.Sequential(
            nn.Conv1d(channel_dim, channel_dim // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(channel_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_dim // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(band, max(8, band // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, band // 4), band),
            nn.Sigmoid()
        )
    def forward(self, x):
        bg_prob = self.bg_encoder(x)         # [B,1,L]
        suppress_mask = 1.0 - bg_prob
        x_suppressed = x * suppress_mask
        x_flat = x_suppressed.mean(dim=1)    # [B,L]
        gate = self.residual_gate(x_flat).unsqueeze(1)  # [B,1,L]
        out = x_suppressed * gate + x * (1 - gate)
        return out, bg_prob.squeeze(1)

class SpectralGroupAttention(nn.Module):
    def __init__(self, band=205, group_length=8, channel_dim=128, state_size=64, layer=1,
                 use_wavelet=True, wavelet_level=1, learnable_lp=False, wavelet_name="db4",
                 dropout=0.0, attn_detach=True, lo_scale=1.0, hi_scale=1.0, use_in_norm=True,
                 multi_m=None, m_gate_type="hilo", use_bg_suppress=True):
        super().__init__()
        self.band = band; self.channel_dim = channel_dim
        self.use_wavelet = use_wavelet; self.attn_detach = attn_detach
        self.use_bg_suppress = use_bg_suppress; self._last_debug = {}
        # 目标感知频谱注意力
        self.spectral_attention = SpectralAttention(band, reduction=16, use_target=True)
        # 输入映射 + 归一化
        self.input_proj = nn.Sequential(nn.Conv1d(1, channel_dim, kernel_size=3, padding=1), nn.GELU())
        self.in_norm = nn.InstanceNorm1d(channel_dim, affine=True) if use_in_norm else nn.Identity()
        # 背景抑制
        if use_bg_suppress:
            self.bg_suppressor = BackgroundSuppressor(channel_dim, band)
        # SWT
        if use_wavelet:
            self.swt_levels = nn.ModuleList(
                [DiffSWT1D(channel_dim, wavelet=wavelet_name, level=l + 1,
                           learnable=learnable_lp, lo_scale=lo_scale, hi_scale=hi_scale)
                 for l in range(wavelet_level)]
            )
            self.low_proc = nn.Sequential(
                nn.Conv1d(channel_dim, channel_dim, 3, padding=1, bias=True),
                nn.BatchNorm1d(channel_dim),
                nn.GELU(),
            )
            self.high_proc = nn.Sequential(
                nn.Conv1d(channel_dim, channel_dim * 2, 3, padding=1, bias=True),
                nn.BatchNorm1d(channel_dim * 2), nn.GELU(),
                nn.Conv1d(channel_dim * 2, channel_dim * 2, 3, padding=1, bias=True),
                nn.BatchNorm1d(channel_dim * 2), nn.GELU(),
                nn.Conv1d(channel_dim * 2, channel_dim, 1, bias=True), nn.GELU(),
            )
            self.se = SE1d(channel_dim, r=8)
        # 主干 Mamba
        self.mamba_main = Mamba(seq_len=band, d_model=channel_dim, state_size=state_size, layer=layer)
        # 多 m
        self.multi_m = list(multi_m) if multi_m else [group_length]
        self.m_gate_type = m_gate_type
        self.group_divs = nn.ModuleList(); self.mamba_groups = nn.ModuleList()
        self.heads = nn.ModuleList(); self.seq_lens = []
        for m in self.multi_m:
            stride = max(1, m // 2)
            self.group_divs.append(nn.Sequential(nn.Conv1d(channel_dim, channel_dim, kernel_size=m, stride=stride, padding=0), nn.GELU()))
            seq_len_group = (band - m) // stride + 1
            self.seq_lens.append(seq_len_group)
            self.mamba_groups.append(Mamba(seq_len=seq_len_group, d_model=channel_dim, state_size=state_size, layer=layer))
            head = [nn.Linear(seq_len_group * channel_dim, 256), nn.GELU(),
                    nn.Linear(256, 128), nn.GELU(),
                    nn.Linear(128, band)]
            self.heads.append(nn.Sequential(*head))
        # 门控参数
        K = len(self.multi_m)
        self.gate_a = nn.Parameter(torch.zeros(K))
        self.gate_b = nn.Parameter(torch.zeros(K))
        self.gate_temp = nn.Parameter(torch.ones(1))  # learnable temperature

    @torch.no_grad()
    def _collect_debug(self, low, high, se_w, gate_w=None, bg_prob=None, att_w=None):
        low_e = low.pow(2).mean().sqrt().item()
        high_e = high.pow(2).mean().sqrt().item()
        ratio = high_e / (low_e + 1e-8)
        dbg = {"se_mean": se_w.mean().item(), "se_std": se_w.std().item(), "hi_lo_ratio": ratio}
        if gate_w is not None:
            dbg.update({"gate_max": int(torch.argmax(gate_w).item()),
                        "gate_w_max": float(gate_w.max().item()),
                        "gate_w_mean": [float(v) for v in gate_w.detach().cpu().tolist()]})
        if bg_prob is not None: dbg["bg_prob_mean"] = float(bg_prob.mean().item())
        if att_w is not None:
            aw = att_w.mean(dim=0); topk = torch.topk(aw, k=min(5, aw.numel()))
            dbg["att_top5_idx"] = [int(i) for i in topk.indices.detach().cpu().tolist()]
        self._last_debug = dbg

    def _compute_gate(self, ratio: float, device: torch.device):
        K = len(self.multi_m); a, b = self.gate_a, self.gate_b
        if self.m_gate_type == "hilo":
            r = torch.tensor([ratio], device=device, dtype=a.dtype); logits = a * r + b
        else:
            logits = b
        temp = torch.clamp(self.gate_temp, min=0.33, max=3.0)
        w = torch.softmax(logits / temp, dim=0)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(2) == 1: x = x.squeeze(2)

        # 目标感知频谱注意力
        tgt = getattr(self, "target", None)
        x = self.spectral_attention(x, target_vec=tgt)

        feat = self.in_norm(self.input_proj(x))
        bg_prob = None
        if self.use_bg_suppress:
            feat, bg_prob = self.bg_suppressor(feat)

        # —— 当前样本的 hi/lo 统计 —— #
        if self.use_wavelet:
            low = feat; highs = []
            for swt in self.swt_levels:
                low, high = swt(low); highs.append(high)
            low = self.low_proc(low)
            high_ref = self.high_proc(sum(highs) / len(highs) if highs else feat)
            _, w = self.se(high_ref)
            if self.attn_detach:
                w = w + (w.detach() - w).detach(); high_ref = high_ref * w; se_w = w
            else:
                high_ref, se_w = self.se(high_ref)
            fused = low + high_ref
            # ➜ 用“当前样本”的 hi/lo 比计算门控依据
            ratio_now = (high_ref.pow(2).mean().sqrt() / (low.pow(2).mean().sqrt() + 1e-8)).item()
        else:
            fused = feat
            se_w = torch.full((feat.size(0), self.channel_dim, 1), 0.5, device=feat.device)
            ratio_now = 1.0

        feat = self.mamba_main(fused.permute(0, 2, 1)).permute(0, 2, 1)

        gate_w = self._compute_gate(ratio_now, device=feat.device)  # [K]

        outs = []
        for k, (div, mg, head, Lg) in enumerate(zip(self.group_divs, self.mamba_groups, self.heads, self.seq_lens)):
            gfeat = div(feat).permute(0, 2, 1)
            gfeat = mg(gfeat).reshape(gfeat.size(0), -1)
            outs.append(head(gfeat))  # [B,band]
        y = sum(gate_w[k] * outs[k] for k in range(len(outs)))  # [B,band]
        if x.size(-1) == y.size(-1):
            y = y + x.squeeze(1) * 0.1
        out = y.unsqueeze(1)

        # 调试信息
        att_w = self.spectral_attention.last_att
        if self.use_wavelet:
            self._collect_debug(low, high_ref, se_w, gate_w, bg_prob, att_w)
        else:
            self._last_debug = {"se_mean": float(se_w.mean().item()), "se_std": float(se_w.std().item()),
                                "hi_lo_ratio": 1.0,
                                "gate_max": int(torch.argmax(gate_w).item()),
                                "gate_w_max": float(gate_w.max().item()),
                                "gate_w_mean": [float(v) for v in gate_w.detach().cpu().tolist()],
                                "bg_prob_mean": float(bg_prob.mean().item()) if bg_prob is not None else 0.0,
                                "att_top5_idx": []}
        return out

    @property
    def last_debug(self):
        return getattr(self, "_last_debug", {})
