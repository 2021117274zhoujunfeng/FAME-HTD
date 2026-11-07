from __future__ import annotations

import torch
from einops import rearrange

try:
    import selective_scan_cuda  # type: ignore
except Exception:
    selective_scan_cuda = None


# 如果 CUDA 扩展不可用，则导出 selective_scan_fn=None 以触发上层纯 PyTorch fallback。
if selective_scan_cuda is None:
    selective_scan_fn = None  # type: ignore

else:
    class SelectiveScanFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            u,
            delta,
            A,
            B,
            C,
            D=None,
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ):
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if z is not None and z.stride(-1) != 1:
                z = z.contiguous()

            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(
                u, delta, A, B, C, D, z, delta_bias, delta_softplus
            )
            ctx.delta_softplus = delta_softplus
            ctx.has_z = z is not None
            last_state = x[:, :, -1, 1::2]

            if not ctx.has_z:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
                return out if not return_last_state else (out, last_state)
            else:
                ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
                out_z = rest[0]
                return out_z if not return_last_state else (out_z, last_state)

        @staticmethod
        def backward(ctx, dout, *args):
            if not ctx.has_z:
                u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
                z = None
                out = None
            else:
                u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors

            if dout.stride(-1) != 1:
                dout = dout.contiguous()

            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                z,
                delta_bias,
                dout,
                x,
                out,
                None,
                ctx.delta_softplus,
                False,
            )
            dz = rest[0] if ctx.has_z else None
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (
                du,
                ddelta,
                dA,
                dB,
                dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None,
            )


    def selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    ):
        """
        return_last_state=True 时返回 (out, last_state)
        last_state 形状: (batch, dim, dstate)
        """
        return SelectiveScanFn.apply(
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            delta_softplus,
            return_last_state,
        )
