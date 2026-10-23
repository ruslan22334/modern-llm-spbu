"""Portable Torch CPU reference for Triton-style tiles; NOT a compiler/backend.

Run: python -m emugpu.kernels.cpu_tiles [--trace]
Only PyTorch is required. Inputs are finite, non-empty floating-point tensors.
"""
from __future__ import annotations

import argparse
import torch


def vector_add(x, y, block=16, trace=None):
    """One simulated program per tile; masked lanes never access an invalid index."""
    if x.device.type != "cpu" or y.device.type != "cpu":
        raise ValueError("This reference executes on CPU")
    if x.ndim != 1 or x.shape != y.shape or x.numel() == 0 or block <= 0:
        raise ValueError("Equal non-empty vectors and positive block required")
    if x.dtype != y.dtype or not x.is_floating_point():
        raise ValueError("Matching floating-point tensors required")
    if not isinstance(block, int):
        raise ValueError("Tile block must be an integer")
    out = torch.empty_like(x)
    for pid in range((x.numel() + block - 1) // block):
        offsets = pid * block + torch.arange(block)
        mask = offsets < x.numel()
        tile_x = torch.zeros(block, dtype=x.dtype)
        tile_y = torch.zeros(block, dtype=y.dtype)
        tile_x[mask], tile_y[mask] = x[offsets[mask]], y[offsets[mask]]
        out[offsets[mask]] = (tile_x + tile_y)[mask]
        if trace is not None:
            trace.append({"kernel": "add", "pid": pid,
                          "valid_lanes": int(mask.sum()), "tile_size": block})
    return out


def row_softmax(x, trace=None):
    """Power-of-two row tile, -inf padding, max subtraction and FP32 reduction."""
    if x.device.type != "cpu":
        raise ValueError("This reference executes on CPU")
    if x.ndim != 2 or min(x.shape) <= 0 or not torch.isfinite(x).all():
        raise ValueError("A non-empty finite matrix is required")
    if not x.is_floating_point():
        raise ValueError("Floating-point input required")
    rows, cols = x.shape
    block = 1 << (cols - 1).bit_length()
    out = torch.empty(x.shape, dtype=torch.float32)
    for pid in range(rows):
        offsets = torch.arange(block)
        mask = offsets < cols
        tile = torch.full((block,), -float("inf"), dtype=torch.float32)
        tile[mask] = x[pid, offsets[mask]]
        exp = torch.exp(tile - tile.max())
        prob = exp / exp.sum(dtype=torch.float32)
        out[pid, offsets[mask]] = prob[mask]
        if trace is not None:
            trace.append({"kernel": "softmax", "pid": pid,
                          "valid_lanes": cols, "tile_size": block})
    return out


def tiled_matmul(a, b, bm=16, bn=16, bk=16, trace=None):
    """Simulate an MxN output grid, zero-padded edge tiles, FP32 accumulation.

    Torch indexing respects strides of sliced/transposed inputs. BLAS computes
    each tile dot on CPU; no GPU occupancy, cache, concurrency or Tensor Core
    latency is modeled. The order of FP32 sums may differ from a @ b.
    """
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("This reference executes on CPU")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected [M,K] and [K,N]")
    if a.dtype != b.dtype or not a.is_floating_point():
        raise ValueError("Matching floating-point tensors required")
    if not all(isinstance(tile, int) for tile in (bm, bn, bk)):
        raise ValueError("Tile sizes must be integers")
    m, k = a.shape
    _, n = b.shape
    if min(m, n, k, bm, bn, bk) <= 0:
        raise ValueError("All dimensions and tiles must be positive")
    c = torch.empty((m, n), dtype=torch.float32)
    for pm in range((m + bm - 1) // bm):
        for pn in range((n + bn - 1) // bn):
            mi, ni = pm * bm, pn * bn
            mm, nn = min(bm, m - mi), min(bn, n - ni)
            acc = torch.zeros((bm, bn), dtype=torch.float32)
            for ki in range(0, k, bk):
                kk = min(bk, k - ki)
                at = torch.zeros((bm, bk), dtype=torch.float32)
                bt = torch.zeros((bk, bn), dtype=torch.float32)
                at[:mm, :kk] = a[mi:mi + mm, ki:ki + kk]
                bt[:kk, :nn] = b[ki:ki + kk, ni:ni + nn]
                acc += at @ bt
            c[mi:mi + mm, ni:ni + nn] = acc[:mm, :nn]
            if trace is not None:
                trace.append({"kernel": "matmul", "pid": [pm, pn],
                              "output_valid": [mm, nn],
                              "k_tiles": (k + bk - 1) // bk})
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    trace = [] if args.trace else None
    for size in (1, 17, 37, 65):
        # Explicit non-contiguous vectors; logical addressing still works.
        x = torch.randn(2 * size)[::2]
        y = torch.randn(2 * size)[::2]
        torch.testing.assert_close(vector_add(x, y, trace=trace), x + y, rtol=0, atol=0)
    for shape in ((1, 1), (3, 37), (5, 65)):
        x = (1000 + torch.randn(shape[0], 2 * shape[1]))[:, ::2]
        y = row_softmax(x, trace)
        e = torch.exp(x - x.amax(dim=1, keepdim=True))
        torch.testing.assert_close(y, e / e.sum(dim=1, keepdim=True), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(y.sum(dim=1), torch.ones(shape[0]), rtol=0, atol=2e-7)
    for m, n, k in ((1, 1, 1), (17, 19, 23), (35, 33, 65)):
        a = torch.randn(m, 2 * k)[:, ::2]
        b = torch.randn(n, k).T
        ref = (a.double() @ b.double()).float()
        torch.testing.assert_close(tiled_matmul(a, b, trace=trace), ref, rtol=3e-5, atol=3e-5)
    print("PASS: Torch CPU add, stable softmax, tiled matmul; tails and strides checked")
    if trace is not None:
        for event in trace:
            print(event)
        print("Trace describes logical tiles, not GPU timing.")


if __name__ == "__main__":
    main()
