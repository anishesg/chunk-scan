"""chunk_scan: Fused SSD kernel for Mamba-2 style state space models."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from chunk_scan import _C as _kernel
    _HAS_KERNEL = True
except ImportError:
    _HAS_KERNEL = False


def _ssm_forward_torch(
    x: torch.Tensor,       # (B, T, H, Dh) fp16
    B: torch.Tensor,       # (B, T, N) fp16
    C: torch.Tensor,       # (B, T, N) fp16
    A: torch.Tensor,       # (H, N) fp32
    D: torch.Tensor,       # (H,) fp32
    delta: torch.Tensor,   # (B, T, H) fp32
) -> torch.Tensor:
    """Pure-PyTorch sequential SSM forward pass. Correctness oracle."""
    batch, T, H, Dh = x.shape
    N = B.shape[2]

    x_f   = x.float()     # (B, T, H, Dh)
    B_f   = B.float()     # (B, T, N)
    C_f   = C.float()     # (B, T, N)

    # h: running state (B, H, N, Dh)
    h = torch.zeros(batch, H, N, Dh, device=x.device, dtype=torch.float32)
    ys = []

    for t in range(T):
        # delta_t: (B, H), A: (H, N) -> decay: (B, H, N)
        decay = torch.exp(A.unsqueeze(0) * delta[:, t, :].unsqueeze(-1))  # (B, H, N)
        # B_t: (B, N) -> (B, 1, N, 1) for broadcasting with h (B, H, N, Dh)
        B_t = B_f[:, t, :].unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
        x_t = x_f[:, t, :, :]                            # (B, H, Dh)

        # State update: h = decay * h + B_t * x_t
        decay_e = decay.unsqueeze(-1)                    # (B, H, N, 1)
        x_te    = x_t.unsqueeze(2)                       # (B, H, 1, Dh)
        h       = decay_e * h + B_t * x_te              # (B, H, N, Dh)

        # Output: y = sum_n C[t,n] * h[:, :, n, :] + D * x_t
        C_t = C_f[:, t, :].unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
        y_t = (C_t * h).sum(dim=2)                       # (B, H, Dh)
        y_t = y_t + D.unsqueeze(0).unsqueeze(-1) * x_t  # (B, H, Dh)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)   # (B, T, H, Dh)
    return y.to(x.dtype)


class SSDLayer(nn.Module):
    """Mamba-2 SSD layer with fused CUDA kernel for efficient chunk-wise processing.

    Holds learned parameters A, D (fixed across sequence positions) and linear
    projections for input-dependent B, C, delta. The forward pass dispatches to
    the fused CUDA kernel when available, falling back to a pure-PyTorch
    sequential implementation otherwise.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_heads: int,
        chunk_size: int = 128,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model   = d_model
        self.d_state   = d_state
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.chunk_size = chunk_size

        # Diagonal log-decay rates A (negative for stability)
        # Shape (n_heads, d_state), initialized near -0.5
        self.A_log = nn.Parameter(
            torch.ones(n_heads, d_state) * math.log(0.5)
        )

        # Skip connection (one scalar per head)
        self.D = nn.Parameter(torch.ones(n_heads))

        # Input-dependent projections: B, C -> d_state; delta -> n_heads
        # Applied to the d_model-dimensional hidden state
        self.B_proj     = nn.Linear(d_model, d_state, bias=False)
        self.C_proj     = nn.Linear(d_model, d_state, bias=False)
        self.delta_proj = nn.Linear(d_model, n_heads, bias=True)

        # Initialize delta bias so softplus output covers [dt_min, dt_max]
        dt_init_floor = math.log(math.expm1(dt_min))
        dt_init_ceil  = math.log(math.expm1(dt_max))
        nn.init.uniform_(self.delta_proj.bias, dt_init_floor, dt_init_ceil)

    @classmethod
    def from_config(
        cls,
        d_model: int,
        d_state: int,
        n_heads: int,
        chunk_size: int = 128,
    ) -> "SSDLayer":
        return cls(d_model, d_state, n_heads, chunk_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run SSD forward pass.

        Args:
            hidden: (batch, seq_len, d_model) fp16 or fp32

        Returns:
            output: (batch, seq_len, d_model) same dtype as input
        """
        batch, T, _ = hidden.shape
        H  = self.n_heads
        Dh = self.d_head
        N  = self.d_state

        # Compute input-dependent projections (fp32 for stability)
        hidden_f = hidden.float()
        B_vals   = self.B_proj(hidden_f)                             # (B, T, N)
        C_vals   = self.C_proj(hidden_f)                             # (B, T, N)
        delta    = F.softplus(self.delta_proj(hidden_f))             # (B, T, H), positive

        # Reshape hidden into heads: (B, T, H, Dh)
        x = hidden.view(batch, T, H, Dh)

        A = -torch.exp(self.A_log.float())  # (H, N), negative

        if _HAS_KERNEL and x.is_cuda and T % self.chunk_size == 0:
            # Dispatch to fused CUDA kernel
            x_fp16    = x.half().contiguous()
            B_fp16    = B_vals.half().contiguous()
            C_fp16    = C_vals.half().contiguous()
            A_fp32    = A.contiguous()
            D_fp32    = self.D.float().contiguous()
            delta_fp32 = delta.contiguous()

            y = _kernel.fused_ssd_forward(
                x_fp16, B_fp16, C_fp16, A_fp32, D_fp32, delta_fp32,
                self.chunk_size
            )
        else:
            # Pure-PyTorch fallback (also handles non-divisible seq_len)
            y = _ssm_forward_torch(
                x.half() if x.dtype != torch.float16 else x,
                B_vals.half(), C_vals.half(), A, self.D.float(), delta
            )

        # Reshape back to (B, T, d_model)
        y_out = y.view(batch, T, self.d_model)
        return y_out.to(hidden.dtype)

    def reference_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Always use the sequential reference (for testing)."""
        batch, T, _ = hidden.shape
        H  = self.n_heads
        Dh = self.d_head

        hidden_f = hidden.float()
        B_vals   = self.B_proj(hidden_f)
        C_vals   = self.C_proj(hidden_f)
        delta    = F.softplus(self.delta_proj(hidden_f))
        x        = hidden.view(batch, T, H, Dh)
        A        = -torch.exp(self.A_log.float())

        y = _ssm_forward_torch(
            x.half() if x.dtype != torch.float16 else x,
            B_vals.half(), C_vals.half(), A, self.D.float(), delta
        )
        return y.view(batch, T, self.d_model).to(hidden.dtype)
