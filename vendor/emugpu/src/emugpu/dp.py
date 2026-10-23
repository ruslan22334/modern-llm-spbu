"""Manual synchronous data parallelism: replicated parameters, sharded batches.

Call sync_gradients AFTER backward and BEFORE optimizer.step. There are no
backward hooks, no optimizer wrapper, and no automatic data sharding.
"""
from __future__ import annotations
import torch
from torch import nn
from .runtime import Context, EmulationError


def _module_contract(module, comm, operation):
    if not isinstance(module, nn.Module):
        comm._preflight(operation, local_error='module must be torch.nn.Module')
    parameters=list(module.named_parameters())
    buffers=list(module.named_buffers())
    error=None
    entries=[]
    for kind,values in (('parameter',parameters),('buffer',buffers)):
        for name,tensor in values:
            if tensor.device not in comm.devices: error='module tensors must be on assigned comm.devices'
            entries.append((kind,name,list(tensor.shape),str(tensor.dtype),tensor.requires_grad))
    comm._preflight(operation,local_error=error,structure=entries)
    return parameters,buffers


def broadcast_module(module: nn.Module, comm: Context, src: int=0):
    """Copy parameters and buffers from group-local src to every replica."""
    parameters,buffers=_module_contract(module,comm,'dp_broadcast_module')
    with torch.no_grad():
        for _,tensor in parameters+buffers:
            value=comm.broadcast(tensor if comm.rank==src else None,src=src)
            tensor.copy_(value)
    return module


def sync_gradients(module: nn.Module, comm: Context, local_count: int=1,
                   loss_reduction: str='mean'):
    """Global mean gradients, supporting unequal and empty local batches.

    local_count is the number of loss elements/sample weights represented by
    this rank. Local mean gradients are multiplied by local_count before SUM;
    local SUM gradients are not multiplied. Both are divided by global_count.
    Use a zero differentiable loss for an empty batch. A parameter unused by
    every rank retains grad=None; unused on some ranks contributes zero.
    """
    error=None
    if isinstance(local_count,bool) or not isinstance(local_count,int) or local_count<0:
        error='local_count must be a nonnegative integer'
    if loss_reduction not in ('sum','mean'): error='loss_reduction must be sum or mean'
    comm._preflight('dp_loss_contract',local_error=error,reduction=loss_reduction)
    parameters,_=_module_contract(module,comm,'dp_gradient_structure')
    active=[p for _,p in parameters if p.requires_grad]
    flags=[p.grad is not None for p in active]
    _,all_flags=comm._preflight('dp_gradient_presence',flags)
    count=comm.all_reduce(torch.tensor([local_count],dtype=torch.int64,device=comm.device))
    total=int(count.item())
    if total<=0: raise EmulationError('global loss count must be positive')
    with torch.no_grad():
        for i,p in enumerate(active):
            if not any(f[i] for f in all_flags): continue
            grad=torch.zeros_like(p) if p.grad is None else p.grad
            # No contribution for an empty batch, even if a caller supplied
            # a nonzero synthetic gradient with loss_reduction='sum'.
            factor=local_count if loss_reduction=='mean' else int(local_count>0)
            reduced=comm.all_reduce(grad*factor)/total
            if p.grad is None: p.grad=reduced
            else: p.grad.copy_(reduced)
    return total


class DataParallel(nn.Module):
    """Transparent manual DP wrapper; use optimizer over .parameters().

    Construction broadcasts parameters/buffers from src. Forward simply calls
    the local module; invoke .sync_gradients after backward on all group ranks.
    Batch-dependent buffers are not automatically synchronized each forward.
    """
    def __init__(self,module: nn.Module,comm: Context,src: int=0):
        super().__init__()
        self.module,self.comm=module,comm
        broadcast_module(module,comm,src)

    def forward(self,*args,**kwargs):
        return self.module(*args,**kwargs)

    def sync_gradients(self,local_count: int=1,loss_reduction: str='mean'):
        return sync_gradients(self.module,self.comm,local_count,loss_reduction)
