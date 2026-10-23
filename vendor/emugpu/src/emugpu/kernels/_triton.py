"""Actual optional Triton kernels, also suitable for supported CPU interpreter.

Import does not require Triton; calling a wrapper does. FP32 output matmul uses
FP32 accumulation. The simple 2D grid prioritizes clear teaching over speed.
"""
from __future__ import annotations

import os

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def add_kernel(X, Y, Z, N, B: tl.constexpr):
        offsets = tl.program_id(0) * B + tl.arange(0, B)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0)
        y = tl.load(Y + offsets, mask=mask, other=0)
        tl.store(Z + offsets, x + y, mask=mask)

    @triton.jit
    def softmax_kernel(X, Y, N, sx0, sx1, sy0, sy1, B: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, B)
        x = tl.load(X + row * sx0 + col * sx1,
                    mask=col < N, other=-float("inf")).to(tl.float32)
        exp = tl.exp(x - tl.max(x, axis=0))
        out = exp / tl.sum(exp, axis=0)
        tl.store(Y + row * sy0 + col * sy1, out, mask=col < N)

    @triton.jit
    def matmul_kernel(A, B, C, M, N, K,
                      sam, sak, sbk, sbn, scm, scn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        mi = tl.program_id(0) * BM + tl.arange(0, BM)
        ni = tl.program_id(1) * BN + tl.arange(0, BN)
        ki = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for kb in range(tl.cdiv(K, BK)):
            kk = kb * BK + ki
            ap = A + mi[:, None] * sam + kk[None, :] * sak
            bp = B + kk[:, None] * sbk + ni[None, :] * sbn
            av = tl.load(ap, (mi[:, None] < M) & (kk[None, :] < K), other=0)
            bv = tl.load(bp, (kk[:, None] < K) & (ni[None, :] < N), other=0)
            acc = tl.dot(av, bv, acc, input_precision="ieee")
        cp = C + mi[:, None] * scm + ni[None, :] * scn
        tl.store(cp, acc, (mi[:, None] < M) & (ni[None, :] < N))

    # Autotuner construction may initialize a GPU driver. Construct it only
    # for an explicit tuned call, so ordinary imports stay driver-independent.
    tuned_add_kernel = None


def require_triton():
    if not TRITON_AVAILABLE:
        raise RuntimeError("Optional Triton is unavailable. Install emugpu[triton] on a supported CUDA host, "
                           "or use emugpu.kernels.cpu_tiles for the portable Torch CPU track.")


def _require_device(*tensors):
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    expected = "cpu" if interpret else "cuda"
    if any(tensor.device.type != expected for tensor in tensors):
        raise ValueError("Triton kernels require CUDA tensors. For the installed CPU interpreter set "
                         "TRITON_INTERPRET=1 before Python starts; portable Torch references are in "
                         "emugpu.kernels.cpu_tiles")


def get_tuned_add_kernel():
    require_triton()
    if os.environ.get("TRITON_INTERPRET") == "1":
        raise ValueError("Autotune is disabled in the CPU interpreter")
    global tuned_add_kernel
    if tuned_add_kernel is None:
        tuned_add_kernel = triton.autotune(configs=[
            triton.Config({"B": 128}, num_warps=4),
            triton.Config({"B": 256}, num_warps=4),
            triton.Config({"B": 512}, num_warps=8),
        ], key=["N"])(add_kernel)
    return tuned_add_kernel


def vector_add(x, y, tuned=False):
    require_triton()
    import torch
    if x.ndim != 1 or x.shape != y.shape or x.device != y.device or x.dtype != y.dtype:
        raise ValueError("Equal 1D shapes, devices and dtypes required")
    if not x.is_contiguous() or not y.is_contiguous() or x.numel() == 0:
        raise ValueError("Non-empty contiguous vectors required by this wrapper")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Expected FP16/BF16/FP32 input")
    _require_device(x, y)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["B"]),)
    if tuned:
        get_tuned_add_kernel()[grid](x, y, out, x.numel())
    else:
        add_kernel[grid](x, y, out, x.numel(), B=256, num_warps=4)
    return out


def row_softmax(x):
    require_triton()
    import torch
    if x.ndim != 2 or min(x.shape) <= 0 or x.shape[1] > 8192:
        raise ValueError("Non-empty [rows,cols] input; educational limit cols <= 8192")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("This example accepts FP16/FP32; interpreter track uses FP32")
    _require_device(x)
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(x.shape[1])
    softmax_kernel[(x.shape[0],)](x, out, x.shape[1], *x.stride(), *out.stride(),
                                 B=block, num_warps=4 if block <= 2048 else 8)
    return out


def tiled_matmul(a, b):
    require_triton()
    import torch
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected [M,K] and [K,N]")
    if a.device != b.device or a.dtype != b.dtype or a.dtype not in (torch.float16, torch.float32):
        raise ValueError("Equal devices and FP16/FP32 dtype required")
    m, k = a.shape
    _, n = b.shape
    if min(m, n, k) <= 0 or min(*a.stride(), *b.stride()) <= 0:
        raise ValueError("Positive shapes and strides required")
    _require_device(a, b)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    tiled_grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    matmul_kernel[tiled_grid](a, b, out, m, n, k, *a.stride(), *b.stride(),
                             *out.stride(), BM=32, BN=32, BK=32, num_warps=4)
    return out


# Small API for the distributed practice: each process supplies local tensors.
add = vector_add
softmax = row_softmax
matmul = tiled_matmul
