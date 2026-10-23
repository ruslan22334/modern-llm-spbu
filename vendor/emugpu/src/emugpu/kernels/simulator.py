"""Torch CPU tile simulator for Triton program_id/grid/mask semantics.

These loops are intentionally sequential. CUDA/Triton execution lives in the
separate real Triton examples and is selected explicitly with --triton.
"""
from dataclasses import dataclass
from typing import Callable

import torch


def cdiv(n: int, block: int) -> int:
    if n < 0 or block <= 0:
        raise ValueError("cdiv needs n >= 0 and block > 0")
    return (n + block - 1) // block


@dataclass(frozen=True)
class Program:
    pid: int
    num_programs: int

    def program_id(self, axis: int = 0) -> int:
        if axis != 0:
            raise ValueError("this teaching simulator uses a one-dimensional grid")
        return self.pid


def launch(grid: int, kernel: Callable, *args, **kwargs) -> None:
    if not isinstance(grid, int) or grid < 0:
        raise ValueError("grid must be a nonnegative integer")
    for pid in range(grid):
        kernel(Program(pid, grid), *args, **kwargs)


def load(pointer: torch.Tensor, offsets: torch.Tensor, mask: torch.Tensor | None = None,
         other: float = 0) -> torch.Tensor:
    if pointer.ndim != 1 or offsets.dtype != torch.int64:
        raise ValueError("load needs flat tensor and int64 offsets")
    mask = torch.ones_like(offsets, dtype=torch.bool) if mask is None else mask.expand_as(offsets)
    if torch.any((offsets[mask] < 0) | (offsets[mask] >= pointer.numel())):
        raise IndexError("unmasked load outside tensor")
    out = torch.full(offsets.shape, other, dtype=pointer.dtype, device=pointer.device)
    out[mask] = pointer[offsets[mask]]
    return out


def store(pointer: torch.Tensor, offsets: torch.Tensor, values: torch.Tensor,
          mask: torch.Tensor | None = None) -> None:
    if pointer.ndim != 1 or offsets.dtype != torch.int64:
        raise ValueError("store needs flat tensor and int64 offsets")
    mask = torch.ones_like(offsets, dtype=torch.bool) if mask is None else mask.expand_as(offsets)
    if torch.any((offsets[mask] < 0) | (offsets[mask] >= pointer.numel())):
        raise IndexError("unmasked store outside tensor")
    pointer[offsets[mask]] = values.expand_as(offsets)[mask]


def vector_add(x: torch.Tensor, y: torch.Tensor, block: int = 128) -> torch.Tensor:
    if x.ndim != 1 or x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
        raise ValueError("vector_add needs matching one-dimensional tensors")
    out = torch.empty_like(x)

    def kernel(program):
        offsets = program.program_id() * block + torch.arange(block, device=x.device)
        mask = offsets < x.numel()
        store(out, offsets, load(x, offsets, mask) + load(y, offsets, mask), mask)

    launch(cdiv(x.numel(), block), kernel)
    return out


def row_softmax(x: torch.Tensor, block: int | None = None) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] == 0 or not x.is_floating_point():
        raise ValueError("row_softmax needs floating [rows, columns], columns > 0")
    rows, cols = x.shape
    block = (1 << (cols - 1).bit_length()) if block is None else block
    if block < cols:
        raise ValueError("block must cover the complete row")
    flat = x.contiguous().view(-1)
    out = torch.empty_like(flat)

    def kernel(program):
        lanes = torch.arange(block, device=x.device)
        offsets = program.program_id() * cols + lanes
        mask = lanes < cols
        values = load(flat, offsets, mask, other=-float("inf")).float()
        numerator = torch.exp(values - values.max())
        store(out, offsets, numerator / numerator.sum(), mask)

    launch(rows, kernel)
    return out.view_as(x)
