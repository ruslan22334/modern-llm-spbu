import copy
import unittest
import torch
from torch import nn
from emugpu import World, DataParallel, EmulationError


class BranchModel(nn.Module):
    def __init__(self,rank):
        super().__init__()
        self.used=nn.Parameter(torch.tensor([1.+rank],dtype=torch.float64))
        self.conditional=nn.Parameter(torch.tensor([2.+rank],dtype=torch.float64))
        self.never=nn.Parameter(torch.tensor([7.+rank],dtype=torch.float64))
        self.register_buffer('counter',torch.tensor([rank],dtype=torch.int64))

    def forward(self,x,use_conditional=False):
        y=x*self.used
        return y+x*self.conditional if use_conditional else y


def worker(ctx,reduction):
    module=BranchModel(ctx.rank)
    mesh=ctx.mesh((1,ctx.size),('replica','dp'))
    model=DataParallel(module,mesh['dp'].comm)
    assert module.counter.item()==0 and module.never.item()==7.
    reference=copy.deepcopy(module)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.2)
    reference_optimizer=torch.optim.Adam(reference.parameters(),lr=.01,weight_decay=.2)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);reference_optimizer.zero_grad(set_to_none=True)
        # Rank0 owns zero samples; conditional parameter is unused on even ranks.
        count=ctx.rank
        x=torch.arange(1,count+1,dtype=torch.float64)
        loss=model(x,ctx.rank%2==1).square().sum()
        if reduction=='mean':loss=loss/max(1,count)
        loss.backward()
        total=model.sync_gradients(count,loss_reduction=reduction)
        expected=sum(reference(torch.arange(1,r+1,dtype=torch.float64),r%2==1).square().sum()
                     for r in range(ctx.size))/total
        expected.backward()
        for p,q in zip(module.parameters(),reference.parameters()):
            if q.grad is None:assert p.grad is None
            else:torch.testing.assert_close(p.grad,q.grad,rtol=1e-10,atol=1e-10)
        optimizer.step();reference_optimizer.step()
        for p,q in zip(module.parameters(),reference.parameters()):torch.testing.assert_close(p,q)
    return True


def group_worker(ctx):
    mesh=ctx.mesh((2,2),('replica','dp'))
    group=mesh['dp'].comm
    model=DataParallel(nn.Linear(2,1,dtype=torch.float64),group)
    x=torch.tensor([[mesh.get_coordinate()[0]+1.,group.rank+1.]],dtype=torch.float64)
    y=model(x).square().sum();y.backward();model.sync_gradients(1)
    for p in model.parameters():
        all_grads=group.all_gather(p.grad.flatten())
        torch.testing.assert_close(all_grads.chunk(2)[0],all_grads.chunk(2)[1])
    return True


def invalid_count(ctx):
    model=DataParallel(nn.Linear(2,1),ctx)
    model(torch.ones(1,2)).sum().backward()
    try:model.sync_gradients(-1 if ctx.rank==0 else 1)
    except EmulationError:ctx.barrier();return True
    raise AssertionError('invalid count accepted')


class DPTests(unittest.TestCase):
    def test_unequal_empty_and_unused_with_adam(self):
        for reduction in ('sum','mean'):
            self.assertEqual(World(4).run(worker,args=(reduction,)),[True]*4)

    def test_independent_mesh_groups(self):
        self.assertEqual(World(4).run(group_worker),[True]*4)

    def test_asymmetric_invalid_count(self):
        self.assertEqual(World(2).run(invalid_count),[True]*2)

if __name__=='__main__':unittest.main()
