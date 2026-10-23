"""Optional real Triton examples: GPU kernels or installed CPU interpreter.

python -m emugpu.kernels.run --device cuda [--benchmark] [--autotune]
TRITON_INTERPRET=1 python -m emugpu.kernels.run --device cpu
"""
from __future__ import annotations

import argparse
import os
import statistics


def cuda_events_ms(fn, repeats=30):
    """GPU execution time; exclude JIT/autotune, allocations and CPU enqueue cost."""
    import torch
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--kernel", choices=("all", "add", "softmax", "matmul"), default="all")
    args = parser.parse_args()
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if args.device == "cpu" and not interpret:
        parser.error("CPU execution requires TRITON_INTERPRET=1 before Python starts; or use python -m emugpu.kernels.cpu_tiles")
    if args.device == "cuda" and interpret:
        parser.error("Use --device cpu for interpreter; unset TRITON_INTERPRET for compiled GPU kernels")
    if interpret and (args.benchmark or args.autotune):
        parser.error("GPU benchmark and autotune are not valid in the CPU interpreter track")
    from . import _triton as kernels
    if not kernels.TRITON_AVAILABLE:
        parser.exit(2, "Optional Triton is not installed. Install emugpu[triton]. Portable CPU track: python -m emugpu.kernels.cpu_tiles\n")
    try:
        import torch
    except ImportError:
        parser.exit(2, "PyTorch is required for both the CPU reference and optional Triton track.\n")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.exit(2, "No CUDA device detected. Use emugpu.kernels.cpu_tiles or an installed CPU interpreter.\n")
    torch.manual_seed(7)
    device = args.device
    x = torch.randn(1003, device=device, dtype=torch.float32)
    y = torch.randn_like(x)
    if args.kernel in ("all", "add"):
        torch.testing.assert_close(kernels.vector_add(x, y, tuned=args.autotune), x + y, rtol=0, atol=0)
    sx = (torch.randn(5, 74, device=device, dtype=torch.float32) + 1000)[:, ::2]
    if args.kernel in ("all", "softmax"):
        torch.testing.assert_close(kernels.row_softmax(sx), torch.softmax(sx, dim=1), rtol=2e-5, atol=2e-6)
    dtype = torch.float32 if interpret else torch.float16
    a = torch.randn(35, 130, device=device, dtype=dtype)[:, ::2]
    b = torch.randn(33, 65, device=device, dtype=dtype).T
    ref = a.float() @ b.float()
    if args.kernel in ("all", "matmul"):
        out = kernels.tiled_matmul(a, b)
        torch.testing.assert_close(out, ref, rtol=3e-4, atol=3e-4)
    print(f"PASS: optional Triton {args.kernel}; device={device}, interpreter={interpret}")
    if args.benchmark:
        # Preallocated kernel output isolates GPU kernel execution.
        size = 1 << 22
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        y = torch.randn_like(x)
        z = torch.empty_like(x)
        if args.autotune:
            tuned = kernels.get_tuned_add_kernel()
            launch = lambda: tuned[lambda meta: (kernels.triton.cdiv(size, meta["B"]),)](x, y, z, size)
        else:
            launch = lambda: kernels.add_kernel[(kernels.triton.cdiv(size, 256),)](x, y, z, size, B=256, num_warps=4)
        ms = cuda_events_ms(launch)
        useful_bytes = 3 * size * x.element_size()
        print(f"add median={ms:.4f} ms, useful effective bandwidth={useful_bytes / (ms * 1e6):.2f} GB/s")
        print("Effective bandwidth uses ideal 2 reads + 1 write; it is not measured HBM transaction traffic.")


if __name__ == "__main__":
    main()
