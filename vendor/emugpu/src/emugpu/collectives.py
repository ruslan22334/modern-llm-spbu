"""First-order differentiable collectives for independent losses per rank.

For SUM AllReduce both forward AND backward are SUM. No double backward is supported.
"""
import torch
from torch.autograd.function import once_differentiable

class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, comm, axis):
        ctx.comm, ctx.axis = comm, axis
        return comm.all_gather(x, axis=axis)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return ctx.comm.reduce_scatter(grad, axis=ctx.axis), None, None

class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, comm, axis):
        ctx.comm, ctx.axis = comm, axis
        return comm.reduce_scatter(x, axis=axis)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return ctx.comm.all_gather(grad, axis=ctx.axis), None, None

class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, comm):
        ctx.comm = comm
        return comm.all_reduce(x)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return ctx.comm.all_reduce(grad), None

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, comm, *blocks):
        ctx.comm = comm
        ctx.shapes = [(b.shape, b.device, b.dtype) for b in blocks]
        outputs = comm.all_to_all(list(blocks))
        ctx.output_specs = [(o.shape, o.device, o.dtype) for o in outputs]
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        grads = [torch.zeros(shape, device=device, dtype=dtype) if g is None else g.contiguous()
                 for g, (shape, device, dtype) in zip(grads, ctx.output_specs)]
        return (None, *ctx.comm.all_to_all(grads))

class _Ring(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, comm, shift, tag):
        ctx.comm, ctx.shift, ctx.tag = comm, shift, tag
        return comm.send_recv(x, dst=(comm.rank + shift) % comm.size,
                             src=(comm.rank - shift) % comm.size, tag=tag)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        comm, shift = ctx.comm, ctx.shift
        return comm.send_recv(grad.contiguous(), dst=(comm.rank - shift) % comm.size,
                              src=(comm.rank + shift) % comm.size, tag=ctx.tag), None, None, None


def all_gather(x, comm, axis=0):
    return _AllGather.apply(x, comm, axis)

def reduce_scatter(x, comm, axis=0):
    """SUM forward; AllGather backward."""
    return _ReduceScatter.apply(x, comm, axis)

def all_reduce(x, comm):
    """SUM forward; SUM backward for different losses on different ranks."""
    return _AllReduce.apply(x, comm)

def all_to_all(blocks, comm):
    """Variable row counts; reverse exchange of gradients in backward."""
    return list(_AllToAll.apply(comm, *blocks))

def ring_shift(x, comm, shift=1, tag=100):
    """Move to +shift, receive from -shift; backward rotates in reverse."""
    return _Ring.apply(x, comm, shift, tag)
