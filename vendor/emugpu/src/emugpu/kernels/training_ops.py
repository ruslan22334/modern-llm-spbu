"""Optional training-capable ReLU: Triton CUDA forward + Torch backward.

python -m emugpu.kernels.training_ops --device cpu   # derivative math only
python -m emugpu.kernels.training_ops --device cuda  # actual kernel comparison
Finite inputs only. CUDA/Triton execution was not available during preparation.
"""
from __future__ import annotations

import argparse
import torch

from . import _triton as kernels

if kernels.TRITON_AVAILABLE:
    triton, tl = kernels.triton, kernels.tl

    @triton.jit
    def _relu_kernel(X, Y, N, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        x = tl.load(X + i, mask=i < N, other=0)
        y = tl.maximum(x, 0, propagate_nan=tl.PropagateNan.ALL)
        tl.store(Y + i, y, mask=i < N)


def relu_backward(grad_output, x):
    """The chosen derivative at x==0 is zero, matching Torch ReLU for finite x."""
    return grad_output * (x > 0).to(grad_output.dtype)


class _TritonReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        kernels.require_triton()
        if x.device.type != "cuda":
            raise ValueError("training_ops.relu requires CUDA; CPU practice uses torch.relu")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Expected FP16/BF16/FP32 CUDA tensor")
        if x.numel() == 0:
            raise ValueError("Expected a non-empty tensor")
        work = x.contiguous()
        out = torch.empty_like(work)
        n = work.numel()
        _relu_kernel[(kernels.triton.cdiv(n, 256),)](work, out, n, B=256, num_warps=4)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return relu_backward(grad_output, x)


def relu(x):
    """Use in an MLP on local cuda:0; communication is outside this operator.

    Forward is Triton; backward is ordinary differentiable Torch operations.
    This is an educational autograd bridge, not a fused training kernel.
    Its x>0 derivative specification assumes finite inputs.
    """
    return _TritonReLU.apply(x)


def check_backward_cpu():
    for size in (1, 17, 1003):
        x = torch.linspace(-2, 2, size).requires_grad_()
        dy = torch.linspace(0.2, 1.2, size)
        torch.relu(x).backward(dy)
        torch.testing.assert_close(relu_backward(dy, x.detach()), x.grad, rtol=0, atol=0)
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    dy = torch.tensor([2.0, 3.0, 4.0])
    torch.relu(x).backward(dy)
    torch.testing.assert_close(relu_backward(dy, x.detach()), x.grad, rtol=0, atol=0)
    print("PASS: Torch CPU finite-input ReLU backward formula, including x=0")


def check_cuda():
    kernels.require_triton()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")
    torch.manual_seed(7)
    for dtype in (torch.float16, torch.float32):
        for size in (1, 17, 1003):
            # Includes a masked tail, a non-contiguous view, negative, zero, positive.
            base = torch.randn(size * 2, device="cuda", dtype=dtype)
            base[0] = 0
            x = base[::2].detach().requires_grad_()
            ref = x.detach().clone().requires_grad_()
            dy = torch.randn_like(x)
            y = relu(x)
            expected = torch.relu(ref)
            torch.testing.assert_close(y, expected, rtol=0, atol=0)
            y.backward(dy)
            expected.backward(dy)
            torch.testing.assert_close(x.grad, ref.grad, rtol=0, atol=0)
    print("PASS: actual Triton CUDA ReLU forward + Torch autograd backward")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cpu":
        check_backward_cpu()
    else:
        check_cuda()


if __name__ == "__main__":
    main()
