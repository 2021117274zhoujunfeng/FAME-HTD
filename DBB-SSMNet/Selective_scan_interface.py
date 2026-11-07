import torch
from einops import rearrange

"""Selective scan interface.

The original implementation relies on a CUDA extension named `selective_scan_cuda`.
Some environments (e.g., fresh machines, CPU-only boxes, or CI) may not have the
extension compiled/installed, which would make a plain `import` fail and break
the whole project.

To keep the repo runnable out-of-the-box, we:
  1) try to import `selective_scan_cuda`;
  2) if it is unavailable, we expose `selective_scan_fn = None`.

Downstream code (see DPyramidSSM.S6) already checks `selective_scan_fn is None`
and falls back to a pure PyTorch implementation.
"""

try:
    import selective_scan_cuda  # type: ignore
except Exception:
    selective_scan_cuda = None


if selective_scan_cuda is None:
    # CUDA extension is not available.
    selective_scan_fn = None

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
            last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
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
                False,  # option to recompute out_z, not used here
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
        """If `return_last_state` is True, returns (out, last_state).

        last_state has shape (batch, dim, dstate). Note that the gradient of the last
        state is not considered in the backward pass.
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
