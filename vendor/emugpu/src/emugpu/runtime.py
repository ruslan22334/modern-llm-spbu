"""Multi-process GPU-communication semantics, using CPU Gloo transport.

Each rank is an independent OS process with explicit CPU/CUDA placement.
Named mesh groups communicate independently through CPU Gloo, including CUDA
payloads staged through host memory. This is a correctness/debugging emulator, not a GPU
network or throughput simulator. The metadata preflight is debug overhead.
Typed Gloo collectives stage int16 as int32 and restore the original output
dtype, including fixed-width SUM overflow; trace details expose both payload
and transport sizes. Raw-byte P2P/AllToAll can transmit int16 directly.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class EmulationError(RuntimeError):
    """Invalid emulation contract or a failed communication operation."""


class CommunicationTimeout(EmulationError):
    """A Gloo communication or launcher watchdog exceeded its deadline."""


_ALL_TO_ALL_TAG = 1_500_000_002  # Outside all user P2P header/payload tags.

_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16,
           torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool]


def _platform_defaults() -> None:
    # Importing a library must not change the host process environment. Apply
    # loopback defaults only when actually initializing a local Gloo runtime.
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        os.environ.setdefault("USE_LIBUV", "0")


def _device(device: str | torch.device) -> torch.device:
    resolved = torch.device("cuda:0" if str(device) in ("cuda", "cuda:0") else device)
    if resolved.type not in ("cpu", "cuda") or (resolved.type == "cuda" and (resolved.index is None or resolved.index < 0)):
        raise ValueError("choose cpu or cuda:N with a nonnegative device index")
    return resolved


def normalize_devices(size, device="cpu", devices=None):
    """Return rank -> tuple of devices. Indices refer to visible CUDA devices.

    A scalar entry selects one primary device; a list assigns several devices
    for explicit use by user code. Device assignment is placement, not isolation.
    """
    assignments = [device] * size if devices is None else devices
    if not isinstance(assignments, (list, tuple)) or len(assignments) != size:
        raise ValueError("devices must have one entry per rank")
    result = []
    for entry in assignments:
        values = entry if isinstance(entry, (list, tuple)) else [entry]
        if not values:
            raise ValueError("each rank needs at least one device")
        result.append(tuple(_device(value) for value in values))
    return tuple(result)


def _positive_timeout(timeout: float, name: str = "timeout") -> None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _communication(call: Callable, *args, **kwargs):
    """Expose stable library errors rather than backend-specific exceptions."""
    try:
        return call(*args, **kwargs)
    except (RuntimeError, OSError) as exc:
        message = str(exc)
        if "timeout" in message.lower() or "timed out" in message.lower():
            raise CommunicationTimeout(f"Gloo communication timed out: {message}") from exc
        raise EmulationError(f"Gloo communication failed: {message}") from exc


def _tensor(x: torch.Tensor) -> torch.Tensor:
    _descriptor(x)
    return x.detach().to("cpu").contiguous().clone()


def _collective_tensor(x: torch.Tensor) -> torch.Tensor:
    # Gloo's typed AR/AG/broadcast kernels do not support int16. Widening is
    # exact for MAX and SUM; a final cast restores int16 modular overflow. P2P
    # and our A2A use raw bytes and can transmit int16 without this adaptation.
    out = _tensor(x)
    return out.to(torch.int32) if out.dtype == torch.int16 else out


def _transport_details(descriptor: dict) -> dict:
    widened = descriptor["dtype"] == str(torch.int16)
    return {"transport_dtype": str(torch.int32) if widened else descriptor["dtype"],
            "transport_tensor_bytes": descriptor["bytes"] * (2 if widened else 1),
            "tensor_payload_bytes": descriptor["bytes"]}


def _descriptor(x: torch.Tensor) -> dict:
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.dtype not in _DTYPES:
        raise EmulationError("communication needs a dense real/integer PyTorch tensor")
    if x.device.type not in ("cpu", "cuda"):
        raise EmulationError("communication tensors must be on CPU or CUDA")
    return {"shape": list(x.shape), "dtype": str(x.dtype), "bytes": x.numel() * x.element_size()}


def _describe(x: Any) -> tuple[dict | None, str | None]:
    try:
        return _descriptor(x), None
    except EmulationError as exc:
        return None, str(exc)


def _axis(axis: int, ndim: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int) or ndim == 0 or not -ndim <= axis < ndim:
        raise EmulationError(f"invalid axis={axis} for ndim={ndim}")
    return axis % ndim


def _tag(tag: int) -> int:
    if isinstance(tag, bool) or not isinstance(tag, int) or not 0 <= tag <= 500_000_000:
        raise EmulationError("tag must be an integer in [0, 500000000]")
    return tag * 3


def _header(x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 16:
        raise EmulationError("P2P teaching headers support at most 16 dimensions")
    header = torch.full((18,), -1, dtype=torch.int64)
    header[0], header[1] = _DTYPES.index(x.dtype), x.ndim
    header[2:2 + x.ndim] = torch.tensor(x.shape, dtype=torch.int64)
    return header


def _from_header(header: torch.Tensor) -> torch.Tensor:
    code, ndim = int(header[0]), int(header[1])
    if not 0 <= code < len(_DTYPES) or not 0 <= ndim <= 16:
        raise EmulationError("invalid message header")
    shape = tuple(int(v) for v in header[2:2 + ndim])
    if any(v < 0 for v in shape):
        raise EmulationError("invalid message shape")
    return torch.empty(shape, dtype=_DTYPES[code])


class Work:
    """Own pending P2P buffers. Always wait before exiting a worker."""
    def __init__(self, finish, buffers):
        self._finish, self._buffers = finish, buffers
        self._done, self._result = False, None

    def wait(self):
        if not self._done:
            self._result = self._finish()
            self._done = True
            self._buffers = None
            self._finish = None
        return self._result


class Context:
    """Communication context for a Gloo group, with logical tracing.

    Raw primitives return detached values; use :mod:`emugpu.collectives`
    for first-order differentiable operations. All ranks must call collectives in the same order. A
    directly constructed Context borrows the process group and never destroys it.
    """

    def __init__(self, device: str = "cpu", *, owns_process_group: bool = False,
                 group=None, ranks=None, trace=None, devices=None, timeout=30, subgroups=None):
        if not dist.is_initialized():
            raise EmulationError("initialize the Gloo process group first")
        if str(dist.get_backend(group)) != "gloo":
            raise EmulationError("emugpu requires a Gloo process group")
        self.group = dist.group.WORLD if group is None else group
        self.global_rank = dist.get_rank()
        self._backend_rank, self.size = dist.get_rank(self.group), dist.get_world_size(self.group)
        self.rank = self._backend_rank
        if self.rank < 0 or self.size < 1:
            raise EmulationError("this rank is not a member of the communication group")
        self.ranks = tuple(range(self.size)) if ranks is None else tuple(ranks)
        if self.global_rank in self.ranks:
            self.rank = self.ranks.index(self.global_rank)
        if len(self.ranks) != self.size or self.ranks[self.rank] != self.global_rank:
            raise EmulationError("group ranks must be ordered by group-local rank")
        self.device = _device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA mode requested but CUDA is unavailable")
            torch.cuda.set_device(self.device)
        self.devices = tuple(_device(d) for d in (devices or [self.device]))
        self.trace: list[dict] = [] if trace is None else trace
        self._step = 0
        self._owns_process_group = owns_process_group
        self._process_group = dist.group.WORLD
        self._closed = False
        self.timeout = timeout
        self._subgroups = [] if subgroups is None else subgroups
        self._owns_subgroups = self.group is dist.group.WORLD

    def mesh(self, shape, names, *, ranks=None):
        """Collectively build a named mesh of all world ranks."""
        from .mesh import DeviceMesh
        return DeviceMesh(self, shape, names, ranks=ranks)

    def new_group(self, ranks):
        """All world ranks call this in the same order. Nonmembers get None."""
        members = tuple(ranks)
        valid = (members and all(isinstance(r, int) and not isinstance(r, bool)
                    and 0 <= r < dist.get_world_size() for r in members)
                 and len(set(members)) == len(members))
        self._preflight("new_group", local_error=None if valid else "invalid global group ranks",
                        ranks=members if valid else ())
        if self.group is not dist.group.WORLD:
            raise EmulationError("create groups from the world context")
        group = _communication(dist.new_group, list(members), backend="gloo",
                               timeout=timedelta(seconds=self.timeout))
        if self.global_rank not in members:
            return None
        self._subgroups.append(group)
        return Context(str(self.device), group=group, ranks=members,
                       trace=self.trace, devices=self.devices, timeout=self.timeout,
                       subgroups=self._subgroups)

    def __enter__(self) -> "Context":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed or not dist.is_initialized() or dist.group.WORLD is not self._process_group:
            raise EmulationError("communication context is closed")

    def _peer(self, rank: int) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < self.size:
            raise EmulationError(f"invalid peer {rank}; world size={self.size}")

    def _preflight(self, op: str, descriptor: Any = None, *, local_error: str | None = None,
                   **metadata) -> tuple[int, list]:
        self._ensure_open()
        step, self._step = self._step, self._step + 1
        # Invalid arguments themselves may not be picklable (e.g. a lambda in
        # place of axis). Exchange the explanatory error, not the bad object.
        local = {"op": op, "metadata": metadata if local_error is None else {},
                 "descriptor": descriptor, "error": local_error}
        participants = [None] * self.size
        _communication(dist.all_gather_object, participants, local, group=self.group)
        backend_ranks = sorted(self.ranks)
        participants = [participants[backend_ranks.index(r)] for r in self.ranks]
        for rank, entry in enumerate(participants):
            if entry["error"] is not None:
                raise EmulationError(f"invalid collective input at step {step}, rank {rank}: {entry['error']}")
        expected = (participants[0]["op"], participants[0]["metadata"])
        for rank, entry in enumerate(participants):
            if (entry["op"], entry["metadata"]) != expected:
                raise EmulationError(
                    f"collective order/arguments mismatch at step {step}: "
                    f"rank 0={expected}, rank {rank}={(entry['op'], entry['metadata'])}"
                )
        return step, [entry["descriptor"] for entry in participants]

    @staticmethod
    def _same(descriptors: list[dict]) -> None:
        first = descriptors[0]
        for rank, entry in enumerate(descriptors[1:], 1):
            if entry["shape"] != first["shape"] or entry["dtype"] != first["dtype"]:
                raise EmulationError(f"shape/dtype mismatch: rank 0={first}; rank {rank}={entry}")

    def _record(self, op: str, step: int, descriptor: Any, sent: float, received: float,
                model: str, **details) -> None:
        shape = descriptor.get("shape") if isinstance(descriptor, dict) else descriptor
        dtype = descriptor.get("dtype") if isinstance(descriptor, dict) else details.pop("dtype", None)
        self.trace.append({"rank": self.global_rank, "group_rank": self.rank, "step": step, "op": op, "start": step,
                           "end": step + 1, "group": list(self.ranks), "shape": shape,
                           "dtype": dtype, "sent_bytes": sent, "received_bytes": received,
                           "byte_model": model + "; excludes debug metadata, CPU staging and protocol overhead",
                           "details": details})

    def all_reduce(self, x: torch.Tensor, op: str = "sum") -> torch.Tensor:
        desc, error = _describe(x)
        if not isinstance(op, str) or op not in ("sum", "mean", "max"):
            error = error or "choose sum, mean, or max"
        if desc is not None and isinstance(op, str) and op == "mean" and not x.is_floating_point():
            error = error or "mean reduction requires floating-point input"
        step, descriptors = self._preflight("all_reduce", desc, local_error=error, reduction=op)
        self._same(descriptors)
        out = _collective_tensor(x)
        _communication(dist.all_reduce, out, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM, group=self.group)
        if op == "mean":
            out /= self.size
        transport = _transport_details(desc)
        traffic = 2 * (self.size - 1) / self.size * transport["transport_tensor_bytes"]
        self._record("all_reduce", step, desc, traffic, traffic,
                     "reference ideal ring on transport dtype: 2*(p-1)/p*N bytes/rank; Gloo may use another algorithm",
                     reduction=op, **transport)
        return out.to(device=x.device, dtype=x.dtype)

    def all_gather(self, x: torch.Tensor, axis: int = 0) -> torch.Tensor:
        desc, error = _describe(x)
        if desc is not None:
            try:
                _axis(axis, x.ndim)
            except EmulationError as exc:
                error = error or str(exc)
        step, descriptors = self._preflight("all_gather", desc, local_error=error, axis=axis)
        self._same(descriptors)
        axis = _axis(axis, x.ndim)
        inp = _collective_tensor(x)
        blocks = [torch.empty_like(inp) for _ in range(self.size)]
        _communication(dist.all_gather, blocks, inp, group=self.group)
        backend_ranks = sorted(self.ranks)
        blocks = [blocks[backend_ranks.index(r)] for r in self.ranks]
        transport = _transport_details(desc)
        traffic = (self.size - 1) * transport["transport_tensor_bytes"]
        self._record("all_gather", step, desc, traffic, traffic,
                     "reference ideal ring on transport dtype: (p-1)*local_bytes", axis=axis, **transport)
        return torch.cat(blocks, dim=axis).to(device=x.device, dtype=x.dtype)

    def reduce_scatter(self, x: torch.Tensor, op: str = "sum", axis: int = 0) -> torch.Tensor:
        desc, error = _describe(x)
        if desc is not None:
            try:
                normalized_axis = _axis(axis, x.ndim)
                if x.shape[normalized_axis] % self.size:
                    raise EmulationError(f"dimension {x.shape[normalized_axis]} must be divisible by {self.size}")
            except EmulationError as exc:
                error = error or str(exc)
        if not isinstance(op, str) or op not in ("sum", "mean", "max") or (desc is not None and op == "mean" and not x.is_floating_point()):
            error = error or "invalid reduce_scatter reduction/dtype"
        step, descriptors = self._preflight("reduce_scatter", desc, local_error=error, reduction=op, axis=axis)
        self._same(descriptors)
        axis = _axis(axis, x.ndim)
        # Portable Gloo fallback: AR + local slice. Semantic RS is exact; actual
        # transport sends AR payload, so BOTH actual/reference models are stated.
        full = _collective_tensor(x)
        _communication(dist.all_reduce, full, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM, group=self.group)
        if op == "mean":
            full /= self.size
        # Tensor.chunk returns fewer than p chunks for an empty dimension.
        width = full.shape[axis] // self.size
        out = full.narrow(axis, self.rank * width, width).contiguous()
        transport = _transport_details(desc)
        ideal = (self.size - 1) / self.size * transport["transport_tensor_bytes"]
        self._record("reduce_scatter", step, desc, ideal, ideal,
                     "reference ideal RS ring on transport dtype: (p-1)/p*N; implementation uses Gloo AR + slice",
                     reduction=op, axis=axis, implementation="all_reduce_then_slice",
                     implementation_reference_sent_bytes=2 * ideal, **transport)
        return out.to(device=x.device, dtype=x.dtype)

    def broadcast(self, x: torch.Tensor | None = None, src: int = 0) -> torch.Tensor:
        desc, error = (None, None) if x is None else _describe(x)
        try:
            self._peer(src)
        except EmulationError as exc:
            error = error or str(exc)
        step, descriptors = self._preflight("broadcast", desc, local_error=error, source=src)
        source = descriptors[src]
        if source is None:
            raise EmulationError("broadcast source must provide a tensor")
        for entry in descriptors:
            if entry is not None and (entry["shape"] != source["shape"] or entry["dtype"] != source["dtype"]):
                raise EmulationError("broadcast shape/dtype mismatch")
        dtype = next(dtype for dtype in _DTYPES if str(dtype) == source["dtype"])
        transport = _transport_details(source)
        transport_dtype = torch.int32 if dtype == torch.int16 else dtype
        out = _collective_tensor(x) if self.rank == src else torch.empty(source["shape"], dtype=transport_dtype)
        _communication(dist.broadcast, out, self.ranks[src], group=self.group)
        self._record("broadcast", step, source,
                     (self.size - 1) * transport["transport_tensor_bytes"] if self.rank == src else 0,
                     transport["transport_tensor_bytes"] if self.rank != src else 0,
                     "reference direct source-to-peer transport payload", source=src, **transport)
        return out.to(device=self.device if x is None else x.device, dtype=dtype)

    def all_to_all(self, blocks: list[torch.Tensor]) -> list[torch.Tensor]:
        # Length check travels in the common preflight, so every rank receives
        # the same explanatory error before data communication starts.
        desc, error = [], None
        if not isinstance(blocks, (list, tuple)):
            error = "all_to_all needs a list or tuple of destination tensors"
        else:
            for block in blocks:
                entry, block_error = _describe(block)
                desc.append(entry)
                error = error or block_error
            if len(blocks) != self.size:
                error = error or f"all_to_all needs {self.size} destination blocks per rank"
            if error is None and any(x.device != blocks[0].device for x in blocks):
                error = "all_to_all local blocks must be on one device"
        step, descriptors = self._preflight("all_to_all", desc, local_error=error)
        if any(len(entry) != self.size for entry in descriptors):
            raise EmulationError(f"all_to_all needs {self.size} destination blocks per rank")
        first = descriptors[0][0]
        if not first["shape"]:
            raise EmulationError("all_to_all needs a leading row-count dimension")
        for rank_blocks in descriptors:
            for entry in rank_blocks:
                if not entry["shape"]:
                    raise EmulationError("all_to_all needs a leading row-count dimension for every block")
                if entry["shape"][1:] != first["shape"][1:] or entry["dtype"] != first["dtype"]:
                    raise EmulationError("all_to_all needs equal trailing shape/dtype; row counts may vary")
        device = blocks[0].device
        incoming = []
        operations = []
        cpu_blocks = [_tensor(x) for x in blocks]
        for src in range(self.size):
            incoming.append(torch.empty(descriptors[src][self.rank]["shape"], dtype=blocks[0].dtype))
        incoming[self.rank].copy_(cpu_blocks[self.rank])
        for peer in range(self.size):
            if peer == self.rank:
                continue
            if cpu_blocks[peer].numel():
                operations.append(dist.P2POp(dist.isend, cpu_blocks[peer], self.ranks[peer], group=self.group, tag=_ALL_TO_ALL_TAG))
            if incoming[peer].numel():
                operations.append(dist.P2POp(dist.irecv, incoming[peer], self.ranks[peer], group=self.group, tag=_ALL_TO_ALL_TAG))
        if operations:
            for request in _communication(dist.batch_isend_irecv, operations):
                _communication(request.wait)
        sent = sum(entry["bytes"] for i, entry in enumerate(desc) if i != self.rank)
        received = sum(descriptors[i][self.rank]["bytes"] for i in range(self.size) if i != self.rank)
        self._record("all_to_all", step, [entry["shape"] for entry in desc], sent, received,
                     "exact off-diagonal payload; implementation batched isend/irecv",
                     dtype=first["dtype"], send_counts=[entry["shape"][0] for entry in desc],
                     receive_counts=[entry[self.rank]["shape"][0] for entry in descriptors])
        return [x.to(device) for x in incoming]

    def reduce(self, x, dst=0, op="sum"):
        """Reduce to a group-local destination (AR fallback)."""
        error = None
        try: self._peer(dst)
        except EmulationError as exc: error = str(exc)
        self._preflight("reduce_destination", local_error=error, destination=dst)
        out = self.all_reduce(x, op=op)
        return out if self.rank == dst else None

    def gather(self, x, dst=0, axis=0):
        """Equal-shaped tensors concatenated at dst (AG fallback)."""
        error = None
        try: self._peer(dst)
        except EmulationError as exc: error = str(exc)
        self._preflight("gather_destination", local_error=error, destination=dst)
        out = self.all_gather(x, axis=axis)
        return out if self.rank == dst else None

    def scatter(self, x=None, src=0, axis=0):
        """Source supplies concatenated equal chunks (broadcast fallback)."""
        self._preflight("scatter_axis", axis=axis)
        full = self.broadcast(x, src=src)
        axis = _axis(axis, full.ndim)
        if full.shape[axis] % self.size:
            raise EmulationError("scatter axis length must divide group size")
        width = full.shape[axis] // self.size
        return full.narrow(axis, self.rank * width, width).contiguous()

    def all_gather_v(self, x):
        """Variable leading row counts; list in logical source rank order."""
        return self.all_to_all([x] * self.size)

    def reduce_scatter_v(self, x, split_sizes, op="sum", axis=0):
        """Reduce full equal inputs, then return variable-size rank slice.

        Portable all_reduce + slice fallback. Every rank supplies the same
        nonnegative split sizes; their sum equals the selected axis length.
        """
        desc, error = _describe(x)
        normalized = None
        if desc is not None:
            try: normalized = _axis(axis, x.ndim)
            except EmulationError as exc: error = str(exc)
        if (not isinstance(split_sizes, (tuple, list)) or len(split_sizes) != self.size or
                any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in split_sizes) or
                (normalized is not None and sum(split_sizes) != x.shape[normalized])):
            error = error or "reduce_scatter_v needs valid splits summing to axis length"
        self._preflight("reduce_scatter_v_contract", desc, local_error=error,
                        splits=split_sizes if error is None else (), reduction=op, axis=axis)
        full = self.all_reduce(x, op=op)
        return full.narrow(normalized, sum(split_sizes[:self.rank]), split_sizes[self.rank]).contiguous()

    def gather_v(self, x, dst=0):
        """Variable leading rows; list in source order at dst, None elsewhere.

        Portable fallback: all_gather_v. All ranks receive temporary data.
        """
        error = None
        try: self._peer(dst)
        except EmulationError as exc: error = str(exc)
        self._preflight("gather_v_destination", local_error=error, destination=dst)
        blocks = self.all_gather_v(x)
        return blocks if self.rank == dst else None

    def scatter_v(self, blocks=None, src=0):
        """Source list of variable-row tensors -> one Tensor per group rank.

        All blocks share trailing shape/dtype; non-sources pass None. Uses
        variable AllToAll with empty outgoing blocks on all non-source ranks.
        """
        error, descriptor = None, None
        try: self._peer(src)
        except EmulationError as exc: error = str(exc)
        if error is None and self.rank == src:
            if not isinstance(blocks, (list, tuple)) or len(blocks) != self.size:
                error = "scatter_v source needs one block per destination"
            else:
                descriptor = []
                for block in blocks:
                    entry, block_error = _describe(block)
                    descriptor.append(entry)
                    error = error or block_error
        elif self.rank != src and blocks is not None:
            error = "scatter_v non-sources must pass None"
        _, descriptors = self._preflight("scatter_v", descriptor, local_error=error, source=src)
        source = descriptors[src]
        first = source[0]
        if not first["shape"] or any(not d["shape"] or d["shape"][1:] != first["shape"][1:] or
                                      d["dtype"] != first["dtype"] for d in source):
            raise EmulationError("scatter_v needs leading row counts and equal trailing shape/dtype")
        dtype = next(d for d in _DTYPES if str(d) == first["dtype"])
        outgoing = list(blocks) if self.rank == src else [
            torch.empty((0, *first["shape"][1:]), dtype=dtype, device=self.device) for _ in range(self.size)]
        return self.all_to_all(outgoing)[src]

    def all_to_all_single(self, x, input_split_sizes=None, output_split_sizes=None):
        """Variable/equal split exchange along axis 0; concatenated output."""
        desc, error = _describe(x)
        splits = input_split_sizes
        if desc is not None:
            if x.ndim == 0: error = "all_to_all_single requires a leading dimension"
            elif splits is None:
                if x.shape[0] % self.size: error = "equal split needs divisible leading dimension"
                else: splits = [x.shape[0] // self.size] * self.size
        if splits is not None and (not isinstance(splits, (list, tuple)) or len(splits) != self.size or
                any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in splits) or
                (desc is not None and x.ndim > 0 and sum(splits) != x.shape[0])):
            error = "invalid input split sizes"
        if output_split_sizes is not None and (not isinstance(output_split_sizes, (list, tuple)) or
                len(output_split_sizes) != self.size or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in output_split_sizes)):
            error = "invalid output split sizes"
        self._preflight("all_to_all_single_contract", local_error=error)
        blocks = self.all_to_all(list(x.split(splits, dim=0)))
        sizes = [b.shape[0] for b in blocks]
        error = None if output_split_sizes is None or list(output_split_sizes) == sizes else "output splits do not match received counts"
        self._preflight("all_to_all_single_output", local_error=error)
        return torch.cat(blocks, dim=0)

    def isend(self, x, dst, tag=0):
        """Post header and payload; retain buffers until Work.wait()."""
        self._ensure_open(); self._peer(dst)
        if dst == self.rank: raise EmulationError("self isend is unsupported")
        base, out = _tag(tag), _tensor(x)
        header = _header(out)
        requests = [_communication(dist.isend, header, self.ranks[dst], group=self.group, tag=base),
                    _communication(dist.isend, out, self.ranks[dst], group=self.group, tag=base + 1)]
        step, self._step = self._step, self._step + 1
        def finish():
            for request in requests: _communication(request.wait)
            self._record("isend", step, _descriptor(out), out.numel()*out.element_size(), 0,
                         "exact P2P payload; asynchronous until wait", peer=dst, tag=tag)
        return Work(finish, (out, header, requests))

    def irecv(self, src, tag=0, shape=None, dtype=None):
        """Post header; wait() receives the announced payload and returns it."""
        self._ensure_open(); self._peer(src)
        if src == self.rank: raise EmulationError("self irecv is unsupported")
        base, header = _tag(tag), torch.empty(18, dtype=torch.int64)
        request = _communication(dist.irecv, header, self.ranks[src], group=self.group, tag=base)
        step, self._step = self._step, self._step + 1
        def finish():
            _communication(request.wait)
            out = _from_header(header)
            _communication(dist.recv, out, self.ranks[src], group=self.group, tag=base + 1)
            if (shape is not None and tuple(shape) != tuple(out.shape)) or (dtype is not None and dtype != out.dtype):
                raise EmulationError("irecv incoming shape/dtype mismatch")
            self._record("irecv", step, _descriptor(out), 0, out.numel()*out.element_size(),
                         "exact P2P payload; asynchronous header", peer=src, tag=tag)
            return out.to(self.device)
        return Work(finish, (header, request))

    def barrier(self) -> None:
        step, _ = self._preflight("barrier")
        _communication(dist.barrier, group=self.group)
        self._record("barrier", step, None, 0, 0, "synchronization only")

    def send(self, x: torch.Tensor, dst: int, tag: int = 0) -> None:
        self._ensure_open()
        self._peer(dst)
        if dst == self.rank:
            raise EmulationError("use a local clone for self send")
        base, out, step = _tag(tag), _tensor(x), self._step
        self._step += 1
        _communication(dist.send, _header(out), self.ranks[dst], group=self.group, tag=base)
        _communication(dist.send, out, self.ranks[dst], group=self.group, tag=base + 1)
        self._record("send", step, _descriptor(out), out.numel() * out.element_size(), 0,
                     "exact P2P payload; blocking send", peer=dst, tag=tag)

    def recv(self, src: int, tag: int = 0, shape: tuple | None = None,
             dtype: torch.dtype | None = None) -> torch.Tensor:
        self._ensure_open()
        self._peer(src)
        if src == self.rank:
            raise EmulationError("use a local clone for self recv")
        base, step = _tag(tag), self._step
        self._step += 1
        header = torch.empty((18,), dtype=torch.int64)
        _communication(dist.recv, header, self.ranks[src], group=self.group, tag=base)
        out = _from_header(header)
        # Drain the announced payload before checking optional expectations:
        # otherwise the sender would remain blocked after a local exception.
        _communication(dist.recv, out, self.ranks[src], group=self.group, tag=base + 1)
        if (shape is not None and tuple(shape) != tuple(out.shape)) or (dtype is not None and dtype != out.dtype):
            raise EmulationError(f"recv expected {shape}/{dtype}, received {tuple(out.shape)}/{out.dtype}")
        self._record("recv", step, _descriptor(out), 0, out.numel() * out.element_size(),
                     "exact P2P payload; blocking recv", peer=src, tag=tag)
        return out.to(self.device)

    def send_recv(self, x: torch.Tensor, dst: int, src: int, tag: int = 0,
                  shape: tuple | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Post send AND receive before waiting; safe for a cyclic ring."""
        self._ensure_open()
        self._peer(dst)
        self._peer(src)
        if dst == self.rank or src == self.rank:
            if dst == src == self.rank:
                _descriptor(x)
                _tag(tag)
                if (shape is not None and tuple(shape) != tuple(x.shape)) or (dtype is not None and dtype != x.dtype):
                    raise EmulationError("send_recv incoming shape/dtype mismatch")
                step, self._step = self._step, self._step + 1
                self._record("send_recv", step, _descriptor(x), 0, 0,
                             "local clone; no transport", dst=dst, src=src, tag=tag,
                             receive_shape=list(x.shape))
                return x.detach().clone()
            raise EmulationError("partial self P2P is unsupported")
        base, out, step = _tag(tag), _tensor(x), self._step
        self._step += 1
        sent_header, received_header = _header(out), torch.empty((18,), dtype=torch.int64)
        ops = [dist.P2POp(dist.isend, sent_header, self.ranks[dst], group=self.group, tag=base),
               dist.P2POp(dist.irecv, received_header, self.ranks[src], group=self.group, tag=base)]
        for request in _communication(dist.batch_isend_irecv, ops):
            _communication(request.wait)
        incoming = _from_header(received_header)
        ops = [dist.P2POp(dist.isend, out, self.ranks[dst], group=self.group, tag=base + 1),
               dist.P2POp(dist.irecv, incoming, self.ranks[src], group=self.group, tag=base + 1)]
        for request in _communication(dist.batch_isend_irecv, ops):
            _communication(request.wait)
        if (shape is not None and tuple(shape) != tuple(incoming.shape)) or (dtype is not None and dtype != incoming.dtype):
            raise EmulationError("send_recv incoming shape/dtype mismatch")
        self._record("send_recv", step, _descriptor(out), out.numel() * out.element_size(),
                     incoming.numel() * incoming.element_size(),
                     "exact P2P payload; post both directions before waiting", dst=dst, src=src, tag=tag,
                     receive_shape=list(incoming.shape))
        return incoming.to(x.device)

    def gather_trace(self) -> list[dict] | None:
        self._preflight("gather_trace")
        traces = [None] * self.size if self.rank == 0 else None
        events = self.trace if self.group is dist.group.WORLD else [e for e in self.trace if tuple(e["group"]) == self.ranks]
        _communication(dist.gather_object, events, traces, dst=self.ranks[0], group=self.group)
        return sorted([event for trace in traces for event in trace], key=lambda e: (e["step"], e["rank"])) if traces else None

    def save_trace(self, path: str | Path, gather: bool = True) -> None:
        events = self.gather_trace() if gather else self.trace
        if events is not None:
            payload = {"schema_version": 1, "world_size": self.size,
                       "time_model": "logical operation slots; never elapsed GPU/CPU time", "events": events}
            Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    def close(self) -> None:
        """Release this context; destroy only the group it initialized."""
        if self._closed:
            return
        self._closed = True
        if dist.is_initialized() and dist.group.WORLD is self._process_group:
            if self._owns_subgroups:
                for group in reversed(self._subgroups):
                    dist.destroy_process_group(group)
                self._subgroups.clear()
            if self._owns_process_group:
                dist.destroy_process_group()


def initialize(device: str = "cpu", timeout: float = 30.0) -> Context:
    """Initialize from torchrun RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT."""
    _positive_timeout(timeout)
    _device(device)
    existing = dist.is_initialized()
    world_size = int(os.environ.get("WORLD_SIZE", str(dist.get_world_size()) if existing else "1"))
    if world_size < 1:
        raise ValueError("world size must be positive")
    if existing:
        if world_size != dist.get_world_size():
            raise EmulationError("WORLD_SIZE does not match the existing process group")
        return Context(device, timeout=timeout)
    _platform_defaults()
    _communication(dist.init_process_group, "gloo", timeout=timedelta(seconds=timeout))
    try:
        return Context(device, owns_process_group=True, timeout=timeout)
    except BaseException:
        dist.destroy_process_group()
        raise


def configure_debugger(rank: int, size: int, base_port: int = 5678,
                       wait_for: str = "none", ready_dir: str | Path | None = None) -> None:
    """Optional rank-specific debugpy attach, before starting any Gloo timeout.

    In wait mode a local file readiness barrier prevents ranks without an attached
    debugger from entering init_process_group early. A paused *initialized* rank
    can still hit the explicitly configured Gloo timeout: increase --timeout.
    """
    _debug_options(size, base_port, wait_for)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
        raise ValueError("debugger rank must belong to the world")
    if wait_for != "none" and ready_dir is None:
        raise ValueError("wait mode needs a unique shared local ready directory")
    try:
        import debugpy
    except ImportError as exc:
        raise RuntimeError("--debug requires optional package debugpy") from exc
    debugpy.listen(("127.0.0.1", base_port + rank))
    print(f"[rank {rank}/{size} pid={os.getpid()}] debugpy attach 127.0.0.1:{base_port + rank}", flush=True)
    if wait_for == "all" or (wait_for == "rank0" and rank == 0):
        debugpy.wait_for_client()
    if wait_for != "none":
        directory = Path(ready_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rank-{rank}.ready").write_text(str(os.getpid()))
        while not all((directory / f"rank-{r}.ready").exists() for r in range(size)):
            time.sleep(0.05)


def _debug_options(size: int, base_port: int, wait_for: str) -> None:
    if wait_for not in ("none", "rank0", "all"):
        raise ValueError("wait_for_debugger must be none, rank0, or all")
    if isinstance(base_port, bool) or not isinstance(base_port, int) or not 1 <= base_port <= 65536 - size:
        raise ValueError("debug_base_port plus world size must fit TCP ports 1–65535")


def _spawn_worker(rank: int, size: int, rendezvous: str, output_dir: str, rank_devices: tuple,
                  timeout: float, worker: Callable, args: tuple, kwargs: dict,
                  debug: bool, debug_base_port: int, wait_for_debugger: str):
    torch.set_num_threads(1)
    _platform_defaults()
    if debug:
        configure_debugger(rank, size, debug_base_port, wait_for_debugger,
                           ready_dir=Path(output_dir) / "debug-ready")
    _communication(dist.init_process_group, "gloo", init_method=rendezvous, rank=rank, world_size=size,
                   timeout=timedelta(seconds=timeout))
    try:
        ctx = Context(str(rank_devices[rank][0]), owns_process_group=True, devices=rank_devices[rank], timeout=timeout)
    except BaseException:
        dist.destroy_process_group()
        raise
    try:
        result = worker(ctx, *args, **kwargs)
        torch.save({"result": result, "trace": ctx.trace}, Path(output_dir) / f"rank-{rank}.pt")
    finally:
        ctx.close()


class World:
    """Local ranks in separate spawned OS processes, with explicit placement.

    Communication uses CPU/Gloo, including in CUDA mode. This models semantics,
    not performance or memory capacity of several physical GPUs. Each rank has
    a separate CUDA context/allocator. Ranks sharing a device share its VRAM.
    """
    def __init__(self, size: int, device: str = "cpu", timeout: float = 30.0,
                 debug: bool = False, debug_base_port: int = 5678,
                 wait_for_debugger: str = "none", run_timeout: float | None = None, *,
                 devices=None):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("world size must be a positive integer")
        _positive_timeout(timeout)
        if run_timeout is not None:
            _positive_timeout(run_timeout, "run_timeout")
        _debug_options(size if debug else 1, debug_base_port, wait_for_debugger)
        if wait_for_debugger != "none" and not debug:
            raise ValueError("wait_for_debugger requires debug=True")
        self.size, self.device, self.timeout = size, str(_device(device)), timeout
        self.rank_devices = normalize_devices(size, device, devices)
        self.devices = self.rank_devices
        for assignment in self.rank_devices:
            for resolved in assignment:
                if resolved.type == "cuda" and (not torch.cuda.is_available() or resolved.index >= torch.cuda.device_count()):
                    raise RuntimeError(f"assigned device {resolved} is unavailable")
        self.debug, self.debug_base_port, self.wait_for_debugger = debug, debug_base_port, wait_for_debugger
        self.run_timeout = run_timeout
        self.trace: list[dict] = []

    def run(self, worker: Callable, args: tuple = (), kwargs: dict | None = None) -> list[Any]:
        """Spawn workers and return CPU-loaded results ordered by rank.

        ``worker(ctx, *args, **kwargs)`` MUST be an importable top-level function,
        and the launching script must use ``if __name__ == '__main__':``. No
        working-directory layout is assumed. Results must be torch-serializable.
        """
        if not callable(worker):
            raise TypeError("worker must be an importable top-level callable")
        if not isinstance(args, tuple):
            raise TypeError("args must be a tuple")
        if kwargs is not None and not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dict or None")
        self.trace = []
        with tempfile.TemporaryDirectory(prefix="emugpu-") as directory:
            rendezvous = "file://" + str(Path(directory) / "rendezvous")
            processes = mp.spawn(_spawn_worker, args=(self.size, rendezvous, directory, self.rank_devices,
                     self.timeout, worker, args, kwargs or {}, self.debug,
                     self.debug_base_port, self.wait_for_debugger), nprocs=self.size, join=False)
            # Gloo bounds communication waits. This separate launcher watchdog
            # also bounds user code that never enters communication. Debug mode
            # permits deliberate pauses; its Gloo timeout still applies.
            budget = self.run_timeout if self.run_timeout is not None else max(30.0, self.timeout * 10)
            deadline = None if self.debug and self.run_timeout is None else time.monotonic() + budget
            try:
                while not processes.join(timeout=0.1):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise CommunicationTimeout(f"rank code exceeded launcher deadline {budget}s")
            except (mp.ProcessRaisedException, mp.ProcessExitedException) as exc:
                message = str(exc)
                if "CommunicationTimeout" in message:
                    raise CommunicationTimeout(f"rank worker communication timed out: {message}") from exc
                raise EmulationError(f"rank worker failed: {message}") from exc
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes.processes:
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
            records = [torch.load(Path(directory) / f"rank-{rank}.pt", weights_only=False,
                                  map_location="cpu") for rank in range(self.size)]
        self.trace = sorted([event for record in records for event in record["trace"]],
                            key=lambda e: (e["step"], e["rank"]))
        return [record["result"] for record in records]

    def save_trace(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"schema_version": 1, "world_size": self.size,
            "time_model": "logical rank-local operation slots; not elapsed GPU/CPU time",
            "events": self.trace}, indent=2) + "\n")


def run(worker: Callable, world_size: int = 4, device: str = "cpu", args: tuple = (),
        kwargs: dict | None = None, timeout: float = 30, *, devices=None) -> list[Any]:
    return World(world_size, device=device, timeout=timeout, devices=devices).run(worker, args=args, kwargs=kwargs)
