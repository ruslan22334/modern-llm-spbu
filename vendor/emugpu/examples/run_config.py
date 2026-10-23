"""Runnable placement/mesh presets and manual DP; no model-parallel solutions."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from emugpu import World, DataParallel
from train_dp import worker as dp_worker


def configurations():
    result = {}
    def add(name, shape, axes, devices, mode='mesh'):
        result[name] = dict(shape=list(shape), axes=list(axes), devices=devices, mode=mode)
    for n in (1, 2, 4, 8, 16, 32):
        add(f'cpu-{n}', (n,), ('workers',), ['cpu'] * n)
    for n in (2, 4, 8, 16):
        add(f'cpu-dp{n}', (n,), ('dp',), ['cpu'] * n, 'dp')
    add('gpu1-single', (1,), ('workers',), ['cuda:0'])
    add('gpu1-dp4', (4,), ('dp',), ['cuda:0'] * 4, 'dp')
    for n in (2, 4, 8):
        add(f'gpu{n}-dp{n}', (n,), ('dp',), [f'cuda:{i}' for i in range(n)], 'dp')
    add('gpu2-dp4-shared', (4,), ('dp',), ['cuda:0', 'cuda:0', 'cuda:1', 'cuda:1'], 'dp')
    for dp, tp in ((1, 4), (4, 2), (2, 4), (8, 2)):
        shape=(dp, tp); size=dp*tp
        add(f'cpu-dp{dp}-tpsp{tp}', shape, ('dp', 'tpsp'), ['cpu'] * size)
        add(f'gpu1-dp{dp}-tpsp{tp}', shape, ('dp', 'tpsp'), ['cuda:0'] * size)
        add(f'gpu{size}-dp{dp}-tpsp{tp}', shape, ('dp', 'tpsp'), [f'cuda:{i}' for i in range(size)])
    add('cpu-mesh-2x2x2x2', (2,2,2,2), ('dp','tp','cp','ep'), ['cpu'] * 16)
    add('cpu-fsdp-axis4', (4,), ('fsdp',), ['cpu'] * 4)
    add('cpu-dp4-two-devices', (4,), ('dp',), [['cpu','cpu'] for _ in range(4)], 'dp-placement')
    add('gpu8-dp4-two-devices', (4,), ('dp',), [[f'cuda:{2*i}',f'cuda:{2*i+1}'] for i in range(4)], 'dp-placement')
    return result


def describe(config):
    shape=config['shape']; size=math.prod(shape)
    if len(config['devices']) != size: raise ValueError('one device assignment per process required')
    coords=list(itertools.product(*(range(d) for d in shape)))
    groups={}
    for axis, name in enumerate(config['axes']):
        buckets={}
        for rank, coord in enumerate(coords):
            key=coord[:axis]+coord[axis+1:]
            buckets.setdefault(key,[]).append(rank)
        groups[name]=list(buckets.values())
    return dict(processes=size, mode=config['mode'], shape=shape, axes=config['axes'],
                ranks=[dict(rank=i,devices=d if isinstance(d,list) else [d],coordinate=list(coords[i]))
                       for i,d in enumerate(config['devices'])], groups=groups)


def mesh_worker(ctx, shape, axes):
    mesh=ctx.mesh(tuple(shape),tuple(axes))
    report=dict(rank=ctx.global_rank,pid=os.getpid(),devices=list(map(str,ctx.devices)),coordinate=mesh.get_coordinate())
    for axis in axes:
        group=mesh[axis].get_group()
        total=int(group.all_reduce(torch.tensor([ctx.global_rank],device=ctx.device)))
        assert total==sum(group.ranks)
        report[axis]=dict(ranks=list(group.ranks),sum=total)
    report['status']='PASS'
    return report


class TwoDeviceMLP(nn.Module):
    """Explicit layer placement: students still implement TP themselves."""
    def __init__(self, devices):
        super().__init__()
        self.first=nn.Linear(4,8).to(devices[0],dtype=torch.float64)
        self.last=nn.Linear(8,3).to(devices[-1],dtype=torch.float64)

    def forward(self, x):
        x=x.to(self.first.weight.device)
        return self.last(torch.relu(self.first(x)).to(self.last.weight.device))


def placement_worker(ctx, steps):
    torch.manual_seed(10+ctx.rank)
    local=TwoDeviceMLP(ctx.devices)
    model=DataParallel(local,ctx.mesh((ctx.size,),('dp',))['dp'].get_group())
    reference=TwoDeviceMLP((ctx.device,))
    reference.load_state_dict(local.state_dict())
    opt=torch.optim.SGD(model.parameters(),lr=.03)
    ref_opt=torch.optim.SGD(reference.parameters(),lr=.03)
    counts=[r+2 for r in range(ctx.size)]; count=counts[ctx.rank]; start=sum(counts[:ctx.rank]); total=sum(counts)
    generator=torch.Generator().manual_seed(123)
    error=0.
    for _ in range(steps):
        x=torch.randn(total,4,generator=generator,dtype=torch.float64)
        y=torch.randn(total,3,generator=generator,dtype=torch.float64)
        opt.zero_grad(); ref_opt.zero_grad()
        out=model(x[start:start+count])
        ((out-y[start:start+count].to(out.device)).square().sum()/count).backward()
        model.sync_gradients(local_count=count)
        expected=reference(x)
        ((expected-y.to(expected.device)).square().sum()/total).backward()
        for p,q in zip(local.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad.to(q.device),q.grad,rtol=1e-8,atol=1e-10)
        opt.step();ref_opt.step()
        for p,q in zip(local.parameters(),reference.parameters()):
            torch.testing.assert_close(p.to(q.device),q,rtol=1e-8,atol=1e-10)
            error=max(error,float((p.detach().to(q.device)-q.detach()).abs().max()))
    return dict(rank=ctx.rank,pid=os.getpid(),devices=list(map(str,ctx.devices)),
                parameter_devices=[str(p.device) for p in local.parameters()],steps=steps,max_error=error,status='PASS')


def main():
    presets=configurations()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',nargs='?',choices=sorted(presets))
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--cpu',action='store_true',help='preserve ranks/mesh/device slots, replace every device with CPU')
    parser.add_argument('--dry-run',action='store_true',help='show assignment/groups without starting processes or CUDA')
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--trace')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--debug-base-port',type=int,default=5678)
    parser.add_argument('--wait-for-debugger',choices=('none','rank0','all'),default='none')
    args=parser.parse_args()
    if args.list:
        for name, config in presets.items():
            print(f"{name:28} processes={math.prod(config['shape']):2} mesh={config['shape']} mode={config['mode']}")
        return
    if not args.config: parser.error('choose a configuration or --list')
    if args.steps<1: parser.error('--steps must be positive')
    if args.wait_for_debugger!='none' and not args.debug: parser.error('--wait-for-debugger requires --debug')
    config=presets[args.config]
    if args.cpu:
        config['devices']=[['cpu']*len(d) if isinstance(d,list) else 'cpu' for d in config['devices']]
    description=describe(config)
    if args.dry_run:
        print(json.dumps(description,indent=2)); return
    world=World(description['processes'],devices=config['devices'],timeout=120,
                debug=args.debug,debug_base_port=args.debug_base_port,wait_for_debugger=args.wait_for_debugger)
    if config['mode']=='dp': results=world.run(dp_worker,kwargs={'steps':args.steps})
    elif config['mode']=='dp-placement': results=world.run(placement_worker,kwargs={'steps':args.steps})
    else: results=world.run(mesh_worker,kwargs={'shape':config['shape'],'axes':config['axes']})
    print(json.dumps(results,indent=2))
    if args.trace: world.save_trace(Path(args.trace))


if __name__=='__main__': main()
