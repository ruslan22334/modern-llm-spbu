"""Actual Torch autograd/SGD DP with unequal batches, checked against dense."""
import argparse
import copy
import json
import os
import torch
from torch import nn
from emugpu import World, DataParallel


def worker(ctx,steps=5):
    torch.manual_seed(10+ctx.rank)  # Deliberately different initialization.
    local=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,3)).to(device=ctx.device,dtype=torch.float64)
    mesh=ctx.mesh((ctx.size,),('dp',))
    model=DataParallel(local,mesh['dp'].get_group())
    reference=copy.deepcopy(local)
    optimizer=torch.optim.SGD(model.parameters(),lr=.03)
    reference_optimizer=torch.optim.SGD(reference.parameters(),lr=.03)
    counts=[rank+2 for rank in range(ctx.size)]
    count=counts[ctx.rank];start=sum(counts[:ctx.rank]);global_count=sum(counts)
    generator=torch.Generator().manual_seed(123)
    max_error=0.
    for step in range(steps):
        x=torch.randn(global_count,4,generator=generator,dtype=torch.float64).to(ctx.device)
        target=torch.randn(global_count,3,generator=generator,dtype=torch.float64).to(ctx.device)
        optimizer.zero_grad();reference_optimizer.zero_grad()
        # Sum over output features, mean over local samples.
        loss=(model(x[start:start+count])-target[start:start+count]).square().sum()/count
        loss.backward()
        model.sync_gradients(local_count=count,loss_reduction='mean')
        expected=(reference(x)-target).square().sum()/global_count
        expected.backward()
        for p,q in zip(local.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-10)
            max_error=max(max_error,float((p.grad-q.grad).abs().max()))
        optimizer.step();reference_optimizer.step()
        for p,q in zip(local.parameters(),reference.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-8,atol=1e-10)
            max_error=max(max_error,float((p-q).abs().max()))
    return dict(rank=ctx.rank,pid=os.getpid(),steps=steps,local_batch=count,global_batch=global_count,
                max_error=max_error,status='PASS',device=str(ctx.device))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--world-size',type=int,default=4)
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--devices',help='JSON device assignment for each rank')
    parser.add_argument('--trace')
    args=parser.parse_args()
    if args.steps<1: parser.error('steps must be positive')
    world=World(args.world_size,device=args.device,devices=json.loads(args.devices) if args.devices else None,timeout=120)
    print(json.dumps(world.run(worker,kwargs={'steps':args.steps}),indent=2))
    if args.trace: world.save_trace(args.trace)
