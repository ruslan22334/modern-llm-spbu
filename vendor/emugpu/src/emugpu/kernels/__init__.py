"""Optional Triton operations and portable Torch tile references.

``add``, ``softmax`` and ``matmul`` launch actual Triton kernels; these
forward-only teaching wrappers do not register autograd. ``relu`` has a
Triton forward and Torch backward. Importing this module never imports
Triton or probes a GPU. Install ``emugpu[triton]`` to call the GPU wrappers.

For CPU tile emulation use ``emugpu.kernels.cpu_tiles`` or ``simulator``.
This models tile indexing and masks, not GPU hardware timing.
"""

from __future__ import annotations

from importlib.util import find_spec

from .simulator import Program, cdiv, launch, load, store, vector_add, row_softmax


def triton_available():
    """Return whether Triton is installed, without importing it or a driver."""
    return find_spec("triton") is not None


def add(x, y, *, tuned=False):
    """Triton vector addition for non-empty contiguous 1D tensors.

    GPU execution is explicit. An installed Triton interpreter also works
    on CPU when ``TRITON_INTERPRET=1`` is set before Python starts.
    """
    from ._triton import vector_add as implementation
    return implementation(x, y, tuned=tuned)


def softmax(x):
    """Triton stable row softmax; FP16/FP32 input and FP32 output."""
    from ._triton import row_softmax as implementation
    return implementation(x)


def matmul(a, b):
    """Triton tiled matmul; FP16/FP32 inputs and FP32 accumulation/output."""
    from ._triton import tiled_matmul as implementation
    return implementation(a, b)


def relu(x):
    """Triton CUDA ReLU forward with Torch backward for finite inputs."""
    from .training_ops import relu as implementation
    return implementation(x)


__all__ = ["add", "softmax", "matmul", "relu", "triton_available",
           "Program", "cdiv", "launch", "load", "store", "vector_add", "row_softmax"]
