"""block-wise parameter sharding using emugpu collectives"""

import math

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.func import functional_call

from hw1.model import LlamaLM


class _ShardedForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, unit):
        ctx.unit = unit
        ctx.save_for_backward(x, shard)
        full = unit.comm.all_gather(shard)
        return unit.call_with_weights(x, full)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, shard = ctx.saved_tensors
        unit = ctx.unit
        # gather again because full weights and the forward graph were released
        full = unit.comm.all_gather(shard).requires_grad_(True)
        with torch.enable_grad():
            local_x = x.detach().requires_grad_(ctx.needs_input_grad[0])
            output = unit.call_with_weights(local_x, full)
            variables = (local_x, full) if ctx.needs_input_grad[0] else (full,)
            grads = torch.autograd.grad(output, variables, grad_output)
        input_grad = grads[0] if ctx.needs_input_grad[0] else None
        # loss is already divided by the global token count, so use SUM
        # clone prevents the Gloo fallback's slice from retaining a full buffer
        shard_grad = unit.comm.reduce_scatter(grads[-1]).clone()
        return input_grad, shard_grad, None


class ShardedUnit(nn.Module):
    """keep one flat shard and reconstruct a module only while computing it"""

    def __init__(self, module, comm, name):
        super().__init__()
        self.comm = comm
        self.name = name
        parameters = list(module.named_parameters())
        if not parameters or list(module.buffers()):
            raise ValueError("sharded units need parameters and no buffers")
        if any(not p.requires_grad for _, p in parameters):
            raise ValueError("this implementation expects trainable parameters")
        self.specs = [(key, p.shape, p.numel()) for key, p in parameters]
        self.numel = sum(count for _, _, count in self.specs)
        self.shard_size = math.ceil(self.numel / comm.size)
        flat = torch.cat([p.detach().reshape(-1) for _, p in parameters])
        flat = comm.broadcast(flat if comm.rank == 0 else None, src=0)
        flat = torch.nn.functional.pad(flat, (0, self.shard_size * comm.size - self.numel))
        start = comm.rank * self.shard_size
        self.shard = nn.Parameter(flat[start : start + self.shard_size].clone())
        # template shapes remain on meta, outside registered optimizer parameters
        object.__setattr__(self, "template", module.to("meta"))

    def call_with_weights(self, x, full):
        parameters = {}
        offset = 0
        for name, shape, count in self.specs:
            parameters[name] = full[offset : offset + count].view(shape)
            offset += count
        return functional_call(self.template, parameters, (x,), strict=True)

    def forward(self, x):
        return _ShardedForward.apply(x, self.shard, self)

    def full_parameters(self, gradient=False):
        """collect full values for verification, called by all ranks"""
        value = self.shard.grad if gradient else self.shard.detach()
        if value is None:
            raise RuntimeError("backward must run before gathering gradients")
        return self.comm.all_gather(value)[: self.numel]


class FSDPLlama(nn.Module):
    """shard each complete decoder block and the input/output layers"""

    def __init__(self, model: LlamaLM, comm):
        super().__init__()
        self.config = model.config
        self.embedding = ShardedUnit(model.token_embedding, comm, "token_embedding")
        self.blocks = nn.ModuleList(
            ShardedUnit(block, comm, f"blocks.{index}")
            for index, block in enumerate(model.blocks)
        )
        self.output = ShardedUnit(
            nn.Sequential(model.final_norm, model.lm_head), comm, "output"
        )

    def forward(self, input_ids):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("input_ids must have shape [batch, tokens] within max_seq_len")
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

    def units(self):
        return [self.embedding, *self.blocks, self.output]

    def sharding_info(self):
        return [
            {"unit": unit.name, "full_numel": unit.numel,
             "shard_numel": unit.shard_size,
             "padding_numel": unit.shard_size * unit.comm.size - unit.numel}
            for unit in self.units()
        ]
