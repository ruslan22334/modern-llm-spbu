"""Triton warm-ups for contiguous float32 inputs"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_single(x, out, value, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + col, tl.load(x + col) + value)


@triton.jit
def _add_block(x, out, value, n, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    data = tl.load(x + col, col < n, other=0)
    tl.store(out + col, data + value, col < n)


@triton.jit
def _outer_single(x, y, out, BC: tl.constexpr, BR: tl.constexpr):
    col = tl.program_id(0) * BC + tl.arange(0, BC)
    row = tl.arange(0, BR)
    values = tl.load(x + col)[None, :] + tl.load(y + row)[:, None]
    tl.store(out + row[:, None] * BC + col[None, :], values)


@triton.jit
def _outer_tile(x, y, out, nc, nr, BC: tl.constexpr, BR: tl.constexpr, MUL_RELU: tl.constexpr):
    col = tl.program_id(0) * BC + tl.arange(0, BC)
    row = tl.program_id(1) * BR + tl.arange(0, BR)
    a = tl.load(x + col, col < nc, other=0)
    b = tl.load(y + row, row < nr, other=0)
    if MUL_RELU:
        values = tl.maximum(b[:, None] * a[None, :], 0.0)
    else:
        values = b[:, None] + a[None, :]
    mask = (row[:, None] < nr) & (col[None, :] < nc)
    tl.store(out + row[:, None] * nc + col[None, :], values, mask)


@triton.jit
def _mul_relu_partials(x, y, dz, partials, nc, nr, tiles_r, BC: tl.constexpr, BR: tl.constexpr):
    tile_r = tl.program_id(1)
    col = tl.program_id(0) * BC + tl.arange(0, BC)
    row = tile_r * BR + tl.arange(0, BR)
    a = tl.load(x + col, col < nc, other=0)
    b = tl.load(y + row, row < nr, other=0)
    mask = (row[:, None] < nr) & (col[None, :] < nc)
    grad = tl.load(dz + row[:, None] * nc + col[None, :], mask, other=0)
    active = b[:, None] * a[None, :] > 0
    dx = tl.sum(tl.where(active, grad * b[:, None], 0.0), axis=0)
    tl.store(partials + col * tiles_r + tile_r, dx, col < nc)


@triton.jit
def _sum_rows(x, out, width, BLOCK: tl.constexpr, TILES: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tile in range(TILES):
        col = tile * BLOCK + lane
        acc += tl.load(x + row * width + col, col < width, other=0)
    tl.store(out + row, tl.sum(acc, axis=0))


@triton.jit
def _softmax_rows(x, out, width, BLOCK: tl.constexpr, TILES: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    log2e: tl.constexpr = 1.4426950408889634
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    for tile in range(TILES):
        col = tile * BLOCK + lane
        logits = tl.load(x + row * width + col, col < width, other=-float("inf"))
        new_max = tl.maximum(maximum, tl.max(logits, axis=0))
        weight = tl.exp2((logits - new_max) * log2e)
        denominator = denominator * tl.exp2((maximum - new_max) * log2e) + tl.sum(weight, axis=0)
        maximum = new_max
    for tile in range(TILES):
        col = tile * BLOCK + lane
        logits = tl.load(x + row * width + col, col < width, other=-float("inf"))
        weight = tl.exp2((logits - maximum) * log2e)
        tl.store(out + row * width + col, weight / denominator, col < width)


@triton.jit
def _flashatt(q, k, v, out, n, BQ: tl.constexpr, BK: tl.constexpr, KEY_TILES: tl.constexpr):
    row = tl.program_id(0) * BQ + tl.arange(0, BQ)
    lane = tl.arange(0, BK)
    query = tl.load(q + row, row < n, other=0)
    log2e: tl.constexpr = 1.4426950408889634
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    denominator = tl.full((BQ,), 0, tl.float32)
    weighted_sum = tl.full((BQ,), 0, tl.float32)
    for tile in range(KEY_TILES):
        col = tile * BK + lane
        key = tl.load(k + col, col < n, other=0)
        value = tl.load(v + col, col < n, other=0)
        score = tl.where(col[None, :] < n, query[:, None] * key[None, :], -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, axis=1))
        scale = tl.exp2((maximum - new_max) * log2e)
        weights = tl.exp2((score - new_max[:, None]) * log2e)
        denominator = denominator * scale + tl.sum(weights, axis=1)
        weighted_sum = weighted_sum * scale + tl.sum(weights * value[None, :], axis=1)
        maximum = new_max
    tl.store(out + row, weighted_sum / denominator, row < n)


def _inputs(ndim, *tensors):
    device = tensors[0].device
    for tensor in tensors:
        if tensor.ndim != ndim or tensor.dtype != torch.float32 or tensor.device != device:
            raise ValueError(f"expected {ndim}D float32 tensors on the same device")
    return tuple(t.contiguous() for t in tensors)


def _power_of_two(n):
    return n > 0 and triton.next_power_of_2(n) == n


def add_triton(x, const_val):
    """task 1: add a scalar with one unmasked block"""
    (x,) = _inputs(1, x)
    n = x.numel()
    if not _power_of_two(n):
        raise ValueError("task 1 requires a positive power-of-two length")
    out = torch.empty_like(x)
    _add_single[(1,)](x, out, const_val, n)
    return out


def add_block_triton(x, const_val):
    """task 2: add a scalar with a masked tail"""
    (x,) = _inputs(1, x)
    n = x.numel()
    out = torch.empty_like(x)
    block = 128
    if n:
        _add_block[(triton.cdiv(n, block),)](x, out, const_val, n, block)
    return out


def add_vec_triton(x, y):
    """task 3: compute an outer sum in one unmasked tile"""
    x, y = _inputs(1, x, y)
    nc, nr = x.numel(), y.numel()
    if not (_power_of_two(nc) and _power_of_two(nr)):
        raise ValueError("task 3 requires positive power-of-two lengths")
    out = x.new_empty((nr, nc))
    _outer_single[(1,)](x, y, out, nc, nr)
    return out


def _outer(x, y, mul_relu):
    x, y = _inputs(1, x, y)
    nc, nr = x.numel(), y.numel()
    out = x.new_empty((nr, nc))
    bc, br = 32, 16
    if nr and nc:
        grid = (triton.cdiv(nc, bc), triton.cdiv(nr, br))
        _outer_tile[grid](x, y, out, nc, nr, bc, br, mul_relu)
    return out


def add_vec_block_triton(x, y):
    """task 4: compute a masked outer sum across a 2d grid"""
    return _outer(x, y, False)


def mul_relu_block_triton(x, y):
    """task 5: multiply and apply relu in one tiled kernel"""
    return _outer(x, y, True)


def mul_relu_block_back_triton(x, y, dz):
    """task 6: reduce the outer product gradient over y to get dx"""
    x, y = _inputs(1, x, y)
    (dz,) = _inputs(2, dz)
    nc, nr = x.numel(), y.numel()
    if dz.device != x.device or dz.shape != (nr, nc):
        raise ValueError("dz must have shape (len(y), len(x)) and share x's device")
    if not nc or not nr:
        return torch.zeros_like(x)
    bc, br = 32, 16
    tiles_r = triton.cdiv(nr, br)
    partials = x.new_empty((nc, tiles_r))
    _mul_relu_partials[(triton.cdiv(nc, bc), tiles_r)](x, y, dz, partials, nc, nr, tiles_r, bc, br)
    # sum contributions from the row tiles without atomic writes to dx
    return sum_triton(partials)


def sum_triton(x):
    """task 7: sum each row using a loop over column tiles"""
    (x,) = _inputs(2, x)
    rows, width = x.shape
    out = x.new_empty((rows,))
    block = 128
    if rows:
        _sum_rows[(rows,)](x, out, width, block, triton.cdiv(width, block))
    return out


def softmax_triton(x):
    """task 8: normalize each row in two tiled passes"""
    (x,) = _inputs(2, x)
    rows, width = x.shape
    out = torch.empty_like(x)
    block = 128
    if rows and width:
        _softmax_rows[(rows,)](x, out, width, block, triton.cdiv(width, block))
    return out


def flashatt_triton(q, k, v):
    """task 9: stream tiles of scalar keys and values to each query block"""
    q, k, v = _inputs(1, q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must have the same length")
    n = q.numel()
    out = torch.empty_like(q)
    bq, bk = 16, 128
    if n:
        _flashatt[(triton.cdiv(n, bq),)](q, k, v, out, n, bq, bk, triton.cdiv(n, bk))
    return out
