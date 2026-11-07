# -*- coding: utf-8 -*-
"""
DPyramidSSM.py

Pyramid-Mamba (State Space Model) backbone for hyperspectral sequence encoding.

说明：
- 本文件原本依赖 `Selective_scan_interface`（CUDA 加速核）与 `causal_conv1d`。
- 为了保证“拿到代码即可运行”，这里做了 try/except：
    * 没有 CUDA kernel 时自动回退到纯 PyTorch scan（速度更慢，但功能正确）
    * causal_conv1d 缺失时也不影响运行

输入/输出：
    x: [B, L, D]
    y: [B, L, D]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

# einops 是常见依赖，但在某些最小环境里可能没有。
# 为了让代码“即拷即跑”，这里做一个非常轻量的 fallback（只覆盖本文件用到的 pattern）。
try:
    from einops import rearrange, repeat  # type: ignore
except Exception:
    def rearrange(x, pattern: str, **axes_lengths):
        """Minimal einops.rearrange fallback for patterns used in this file."""
        if pattern == "b c l -> (b l) c":
            # [B,C,L] -> [B*L,C]
            B, C, L = x.shape
            return x.permute(0, 2, 1).reshape(B * L, C)
        if pattern == "d (b l) -> b d l":
            # [D, B*L] -> [B,D,L]
            d, bl = x.shape
            b = int(axes_lengths.get("b"))
            l = int(axes_lengths.get("l"))
            return x.reshape(d, b, l).permute(1, 0, 2)
        if pattern == "(b l) n -> b n l":
            # [B*L, N] -> [B,N,L]
            bl, n = x.shape
            b = int(axes_lengths.get("b"))
            l = int(axes_lengths.get("l"))
            return x.reshape(b, l, n).permute(0, 2, 1)
        if pattern == "d 1 w -> d w":
            return x.squeeze(1)
        if pattern == "b l c -> b c l":
            return x.permute(0, 2, 1)
        if pattern == "b c l -> b l c":
            return x.permute(0, 2, 1)
        raise ImportError(f"einops is not installed and fallback does not support pattern: {pattern}")

    def repeat(x, pattern: str, **axes_lengths):
        """Minimal einops.repeat fallback for patterns used in this file."""
        if pattern == "n -> d n":
            d = int(axes_lengths.get("d"))
            return x.unsqueeze(0).repeat(d, 1)
        if pattern == "b n l -> b c n l":
            c = int(axes_lengths.get("c"))
            return x.unsqueeze(1).repeat(1, c, 1, 1)
        raise ImportError(f"einops is not installed and fallback does not support pattern: {pattern}")
try:
    from Selective_scan_interface import selective_scan_fn  # optional CUDA kernel
except Exception:
    selective_scan_fn = None  # fallback to pure PyTorch scan

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except Exception:
    causal_conv1d_fn, causal_conv1d_update = None, None


# ---------------- RMSNorm ----------------
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: [B,L,D]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------- Down / Up ----------------
class DownSample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):  # x: [B,C,L]
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, target_len: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.target_len = int(target_len)

    def forward(self, x):  # x: [B,C,L]
        y = self.deconv(x)
        if y.shape[-1] != self.target_len:
            y = F.interpolate(y, size=self.target_len, mode='nearest')
        return y


# ---------------- SSM ----------------
class S6(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_inner: int = 128,
                 dt_rank: Union[str, int] = "auto", dt_min=1e-3, dt_max=1e-1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 use_scan_cuda=True):
        super().__init__()
        # channels == 输入的通道数（本实现里就是各尺度的 proj 宽度 d_inner）
        self.channels = channels
        self.d_state = d_state
        self.d_inner = d_inner
        self.use_scan_cuda = use_scan_cuda

        if dt_rank == "auto":
            self.dt_rank = max(1, math.ceil(channels / 16))
        else:
            self.dt_rank = int(dt_rank)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # init dt_proj
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # A matrix (diagonal parameterization), size [d_inner, d_state]
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

    def forward(self, u):  # u: [B,C,L] with C == channels
        B, C, L = u.shape
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        x_dbl = self.x_proj(rearrange(u, "b c l -> (b l) c"))
        delta, Bmat, Cmat = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj.weight @ delta.t()                     # [d_inner, B*L]
        delta = rearrange(delta, "d (b l) -> b d l", b=B, l=L)      # [B, d_inner, L]
        Bmat  = rearrange(Bmat, "(b l) n -> b n l", b=B, l=L)       # [B, d_state, L]
        Cmat  = rearrange(Cmat, "(b l) n -> b n l", b=B, l=L)       # [B, d_state, L]

        delta_bias = self.dt_proj.bias.float()
        if not self.use_scan_cuda or selective_scan_fn is None:
            u_f = u.float()
            delta_f = F.softplus(delta.float() + delta_bias[..., None])
            x = A.new_zeros((B, C, self.d_state))
            ys = []
            deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta_f, A))
            # broadcast B/C over channel
            B_full = repeat(Bmat, "b n l -> b c n l", c=C)
            C_full = repeat(Cmat, "b n l -> b c n l", c=C)
            for i in range(L):
                x = deltaA[:, :, i] * x + u_f[:, :, i].unsqueeze(-1) * B_full[:, :, :, i]
                y = torch.einsum("bdn,bdn->bd", x, C_full[:, :, :, i])
                ys.append(y)
            y = torch.stack(ys, dim=2)  # [B,C,L]
            return y.to(dtype=u.dtype)
        else:
            # variable B,C handled inside kernel
            y = selective_scan_fn(u, delta, A, Bmat, Cmat, None,
                                  z=None, delta_bias=delta_bias,
                                  delta_softplus=True, return_last_state=False)
            return y.to(dtype=u.dtype)


class SSM_Manipulation(nn.Module):
    def __init__(self, channels: int, state_size: int, proj_width: int,
                 d_conv: int, conv_bias=True, use_causal1D=False, adjust=True):
        super().__init__()
        self.depthwise = nn.Conv1d(proj_width, proj_width, kernel_size=d_conv,
                                   padding=d_conv - 1, groups=proj_width, bias=conv_bias)
        self.ssm = S6(channels=proj_width, d_state=state_size, d_inner=proj_width)
        self.act = nn.SiLU()
        self.use_causal1D = use_causal1D
        self.adjust = nn.Conv1d(proj_width, proj_width, kernel_size=1) if adjust else nn.Identity()

    def forward(self, x):  # x: [B,proj_width,L]
        if self.use_causal1D and causal_conv1d_fn is not None:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.depthwise.weight, "d 1 w -> d w"),
                bias=self.depthwise.bias,
                activation="silu"
            )
        else:
            x = self.act(self.depthwise(x)[..., :x.shape[-1]])
        x = self.ssm(x)
        x = self.adjust(x)
        return x


# -------------- Pyramid Mamba Block --------------
class MambaBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int, state_size: int, expand=2, d_conv=3, bias=False):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = d_model
        self.d_inner = expand * d_model

        self.in_proj  = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.norm = RMSNorm(d_model)

        # compute target lengths for upsampling
        seqs = [self.seq_len]
        d = self.seq_len
        for _ in range(2):
            d = (d - 1) // 2 + 1
            seqs.append(d)

        # SSM per scale
        self.ssm1 = SSM_Manipulation(channels=d_model,       state_size=state_size, proj_width=self.d_inner,     d_conv=d_conv, adjust=True)
        self.ssm2 = SSM_Manipulation(channels=2 * d_model,   state_size=state_size, proj_width=2 * self.d_inner, d_conv=d_conv, adjust=True)
        self.ssm3 = SSM_Manipulation(channels=4 * d_model,   state_size=state_size, proj_width=4 * self.d_inner, d_conv=d_conv, adjust=True)
        self.ssm4 = SSM_Manipulation(channels=8 * d_model,   state_size=state_size, proj_width=8 * self.d_inner, d_conv=d_conv, adjust=False)

        # down/up
        self.down1 = DownSample(self.d_inner)
        self.down2 = DownSample(2 * self.d_inner)
        self.down3 = DownSample(4 * self.d_inner)

        self.up1 = UpSample(8 * self.d_inner, 4 * self.d_inner, seqs[2])
        self.up2 = UpSample(4 * self.d_inner, 2 * self.d_inner, seqs[1])
        self.up3 = UpSample(2 * self.d_inner, 1 * self.d_inner, seqs[0])

        self.merge3 = nn.Conv1d(4 * self.d_inner, 4 * self.d_inner, 1)
        self.merge2 = nn.Conv1d(2 * self.d_inner, 2 * self.d_inner, 1)
        self.merge1 = nn.Conv1d(1 * self.d_inner, 1 * self.d_inner, 1)

        self.gate_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):  # x: [B,L,D]
        B, L, D = x.shape
        pre = x
        x = self.norm(x)
        xz = self.in_proj(x)              # [B,L,2*inner]
        x_feat, z = xz.chunk(2, dim=-1)   # [B,L,inner], [B,L,inner]
        gate = torch.sigmoid(self.gate_alpha * z)

        # to channels-first
        x1 = rearrange(x_feat, "b l c -> b c l")

        # multiscale
        c1 = self.ssm1(x1)                        # [B, 1*inner, L]
        x2 = self.down1(x1)                       # [B, 2*inner, L2]
        c2 = self.ssm2(x2)                        # [B, 2*inner, L2]
        x3 = self.down2(x2)                       # [B, 4*inner, L3]
        c3 = self.ssm3(x3)                        # [B, 4*inner, L3]
        x4 = self.down3(x3)                       # [B, 8*inner, L4]
        c4 = self.ssm4(x4)                        # [B, 8*inner, L4]

        # up & merge with dynamic length align
        u3 = self.up1(c4)                         # -> L3
        if u3.shape[-1] != c3.shape[-1]:
            u3 = F.interpolate(u3, size=c3.shape[-1], mode='nearest')
        u3 = self.merge3(u3 + c3)                 # [B,4*inner,L3]

        u2 = self.up2(u3)                         # -> L2
        if u2.shape[-1] != c2.shape[-1]:
            u2 = F.interpolate(u2, size=c2.shape[-1], mode='nearest')
        u2 = self.merge2(u2 + c2)                 # [B,2*inner,L2]

        u1 = self.up3(u2)                         # -> L
        if u1.shape[-1] != c1.shape[-1]:
            u1 = F.interpolate(u1, size=c1.shape[-1], mode='nearest')
        u1 = self.merge1(u1 + c1)                 # [B,1*inner,L]

        u1 = rearrange(u1, "b c l -> b l c")
        out = self.out_proj(u1 * gate)
        return out + pre


class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, layer=1):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(seq_len, d_model, state_size) for _ in range(layer)])

    def forward(self, x):  # [B,L,D]
        for m in self.layers:
            x = m(x)
        return x
