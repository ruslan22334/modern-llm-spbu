"""compare the nine kernels with the PyTorch formulas from the assignment"""

import os

import torch
import triton

from hw3 import kernels


def check_kernels(device):
    generator = torch.Generator(device="cpu").manual_seed(42)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    def check(name, actual, expected, rtol=1e-5, atol=1e-8):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
        print(f"PASS {name}", flush=True)

    x = rand(256)
    check("1 constant add", kernels.add_triton(x, 1.25), x + 1.25)
    x = rand(257)
    check("2 blocked constant add, tail", kernels.add_block_triton(x, -0.75), x - 0.75)

    x, y = rand(64), rand(32)
    check("3 outer add", kernels.add_vec_triton(x, y), x[None, :] + y[:, None])

    x, y = rand(202)[::2], rand(53)
    check("4 blocked outer add, both tails and strided input",
          kernels.add_vec_block_triton(x, y), x[None, :] + y[:, None])
    x, y = x.clone(), y.clone()
    x[0], y[0] = 0.0, 0.0
    z = (x[None, :] * y[:, None]).relu()
    check("5 fused multiply and relu, both tails", kernels.mul_relu_block_triton(x, y), z)

    dz = rand(x.numel(), y.numel()).T
    ref_x = x.clone().requires_grad_()
    ref_z = (ref_x[None, :] * y[:, None]).relu()
    (dx,) = torch.autograd.grad(ref_z, ref_x, dz)
    check("6 backward dx, both tails and zero inputs",
          kernels.mul_relu_block_back_triton(x, y, dz), dx, rtol=1e-4, atol=1e-5)

    rows = rand(7, 1027)
    check("7 long sum, multiple tiles and a tail", kernels.sum_triton(rows), rows.sum(dim=1))

    logits = rand(7, 1027) * 15.0 + 10000.0
    logits[:, 200] = 10100.0
    logits[:, -1] = 10101.0
    check("8 long softmax, large logits with a late maximum",
          kernels.softmax_triton(logits), logits.softmax(dim=1), rtol=1e-4, atol=1e-6)

    q, k, v = rand(259) * 10, rand(259) * 10, rand(259)
    k[-1] = 35.0
    weights = (q[:, None] * k[None, :]).softmax(dim=1)
    check("9 scalar flashattention, multiple tiles and both tails",
          kernels.flashatt_triton(q, k, v), (weights * v[None, :]).sum(dim=1),
          rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    device = "cpu" if interpret else "cuda"
    if not interpret and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Set TRITON_INTERPRET=1 for a CPU logic check.")
    print(f"torch={torch.__version__}, triton={triton.__version__}, mode={device}", flush=True)
    if device == "cuda":
        print(torch.cuda.get_device_name(), flush=True)
    check_kernels(device)
