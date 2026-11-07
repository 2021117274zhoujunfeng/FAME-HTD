# PyramidSSM.py
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from Selective_scan_interface import selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device="cuda"):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )
        return output


class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, layer=1):
        super(Mamba, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(1, layer + 1):
            self.layers.append(
                MambaBlock(
                    seq_len=seq_len,
                    d_model=d_model,
                    state_size=state_size,
                    expand=1,
                    d_conv=3,
                    bias=False,
                )
            )

    def forward(self, x):
        for ma in self.layers:
            x = ma(x)
        return x


class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, expand=1, d_conv=3, bias=False):
        super(MambaBlock, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_state = state_size
        self.expand = expand
        self.d_conv = d_conv
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        seq_array = [seq_len]
        d = seq_len
        for _ in range(2):
            d = (d + 2 * 1 - 3) // 2 + 1
            seq_array.append(d)

        self.ssm_x1 = SSM_Manipulation(
            seq_len=seq_array[0],
            d_model=self.d_inner,
            state_size=self.d_state,
            d_inner=self.d_inner,
            d_conv=self.d_conv,
            conv_bias=True,
            adjust=True,
        )
        self.ssm_x2 = SSM_Manipulation(
            seq_len=seq_array[1],
            d_model=2 * self.d_inner,
            state_size=self.d_state,
            d_inner=2 * self.d_inner,
            d_conv=self.d_conv,
            conv_bias=True,
            adjust=True,
        )
        self.ssm_x3 = SSM_Manipulation(
            seq_len=seq_array[2],
            d_model=4 * self.d_inner,
            state_size=self.d_state,
            d_inner=4 * self.d_inner,
            d_conv=self.d_conv,
            conv_bias=True,
            adjust=True,
        )
        self.ssm_x4 = SSM_Manipulation(
            seq_len=seq_array[2],
            d_model=8 * self.d_inner,
            state_size=self.d_state,
            d_inner=8 * self.d_inner,
            d_conv=self.d_conv,
            conv_bias=True,
            adjust=False,
        )

        self.down_1 = DownSample(self.d_inner)
        self.down_2 = DownSample(2 * self.d_inner)
        self.down_3 = DownSample(4 * self.d_inner)

        self.up1 = UpSample(8 * self.d_inner, seq_array[2], output_padding=1)
        self.up2 = UpSample(4 * self.d_inner, seq_array[1], output_padding=1)
        self.up3 = UpSample(2 * self.d_inner, seq_array[0], output_padding=1)

        self.norm = RMSNorm(self.d_inner)

        self.wavelet_gate = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 4),
            nn.ReLU(),
            nn.Linear(self.d_inner // 4, self.d_inner),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pre_x = x
        x = self.norm(x)

        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Input sequence length {x.shape[1]} does not match expected seq_len {self.seq_len}"
            )

        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=self.seq_len,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)

        x1 = x
        x1_ssm = self.ssm_x1(x1)
        x2 = self.down_1(x1)
        x2_ssm = self.ssm_x2(x2)
        x3 = self.down_2(x2)
        x3_ssm = self.ssm_x3(x3)
        x4 = self.down_3(x3)
        x4_ssm = self.ssm_x4(x4)

        x3_sup = self.up1(x4_ssm)
        if x3_sup.shape[2] != x3_ssm.shape[2]:
            x3_sup = F.interpolate(
                x3_sup, size=x3_ssm.shape[2], mode="linear", align_corners=False
            )
        x3_sup += x3_ssm
        x2_sup = self.up2(x3_sup)
        if x2_sup.shape[2] != x2_ssm.shape[2]:
            x2_sup = F.interpolate(
                x2_sup, size=x2_ssm.shape[2], mode="linear", align_corners=False
            )
        x2_sup += x2_ssm
        x1_sup = self.up3(x2_sup)
        if x1_sup.shape[2] != x1_ssm.shape[2]:
            x1_sup = F.interpolate(
                x1_sup, size=x1_ssm.shape[2], mode="linear", align_corners=False
            )
        x1_sup += x1_ssm

        wavelet_gate = self.wavelet_gate(rearrange(x1_sup, "b d l -> b l d"))
        wavelet_gate = rearrange(wavelet_gate, "b l d -> b d l")
        x1_sup = x1_sup * wavelet_gate

        x_residual = F.silu(z)
        x_combined = x1_sup * x_residual
        x_combined = rearrange(x_combined, "b d l -> b l d")
        x_out = self.out_proj(x_combined)
        return x_out + pre_x


class DownSample(nn.Module):
    def __init__(self, d_model):
        super(DownSample, self).__init__()
        self.conv_down = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        down_x = self.conv_down(x)
        return down_x


class UpSample(nn.Module):
    def __init__(self, d_model, size, output_padding=0):
        super(UpSample, self).__init__()
        self.conv_up = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_model // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=output_padding,
        )

    def forward(self, x):
        up_x = self.conv_up(x)
        return up_x


class SSM_Manipulation(nn.Module):
    def __init__(
        self, seq_len, d_model, state_size, d_inner, d_conv, conv_bias=True, adjust=True
    ):
        super(SSM_Manipulation, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_state = state_size
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.adjust = adjust

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.act = torch.nn.SiLU()

        self.B = nn.Parameter(torch.empty(self.d_inner, self.d_state))
        nn.init.kaiming_uniform_(self.B)
        self.C = nn.Parameter(torch.empty(self.d_inner, self.d_state))
        nn.init.kaiming_uniform_(self.C)
        self.D = nn.Parameter(torch.ones((self.d_inner)))

        self.ssm = S6(self.d_model, self.d_state, self.d_inner)

    def forward(self, x):
        skip = x
        x = rearrange(x, "b d l -> b l d")

        if self.adjust:
            x = x.permute(0, 2, 1)
            x = self.conv1d(x)[:, :, : self.seq_len]
            x = x.permute(0, 2, 1)

        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.ssm(x)
        x = x * self.D.unsqueeze(0).unsqueeze(2)
        x = x + skip

        return x


class S6(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_inner=16,
        dt_rank="auto",
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        dt_min=0.001,
        dt_init_floor=1e-4,
        use_scan_cuda=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_scan_cuda = use_scan_cuda

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

    def forward(self, u):
        batch, dim, seqlen = u.shape
        A = -torch.exp(self.A_log.float())

        x_dbl = self.x_proj(rearrange(u, "b d l -> (b l) d"))
        delta, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = self.dt_proj.weight @ delta.t()
        delta = rearrange(delta, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        delta_bias = self.dt_proj.bias.float()
        delta_softplus = True
        dtype_in = u.dtype
        if not self.use_scan_cuda:
            u = u.float()
            delta = delta.float()
            if delta_bias is not None:
                delta = delta + delta_bias[..., None].float()
            if delta_softplus:
                delta = F.softplus(delta)
            batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
            is_variable_B = B.dim() >= 3
            is_variable_C = C.dim() >= 3
            if A.is_complex():
                if is_variable_B:
                    B = torch.view_as_complex(
                        rearrange(B.float(), "... (L two) -> ... L two", two=2)
                    )
                if is_variable_C:
                    C = torch.view_as_complex(
                        rearrange(C.float(), "... (L two) -> ... L two", two=2)
                    )
            else:
                B = B.float()
                C = C.float()
            x = A.new_zeros((batch, dim, dstate))
            ys = []
            deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
            if not is_variable_B:
                deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
            else:
                if B.dim() == 3:
                    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
                else:
                    B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                    deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
            if is_variable_C and C.dim() == 4:
                C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
            last_state = None
            for i in range(u.shape[2]):
                x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
                if not is_variable_C:
                    y = torch.einsum("bdn,dn->bd", x, C)
                else:
                    if C.dim() == 3:
                        y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
                    else:
                        y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
                if i == u.shape[2] - 1:
                    last_state = x
                if y.is_complex():
                    y = y.real * 2
                ys.append(y)
            y = torch.stack(ys, dim=2)
            out = y
            out = out.to(dtype=dtype_in)
            return out
        else:
            y = selective_scan_fn(
                u,
                delta,
                A,
                B,
                C,
                None,
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            out = y
            out = out.to(dtype=dtype_in)
            return out
