"""Topology only: DP replicas x model ranks; no model-parallel implementation."""
import argparse
import json
import os
import torch
from emugpu import World


def worker(ctx,dp=4,tpsp=2):
    mesh=ctx.mesh((dp,tpsp),('dp','tpsp'))
    # mesh=ctx.mesh((dp,tpsp),('dp','ep'))
    # mesh=ctx.mesh((dp,tpsp),('dp','ep', 'tpsp'))
    report=dict(rank=ctx.global_rank,pid=os.getpid(),device=str(ctx.device),
                devices=[str(d) for d in ctx.devices],coordinate=mesh.get_coordinate())
    for axis in ('dp','tpsp'):
        group=mesh[axis].get_group()
        x=torch.tensor([ctx.global_rank],device=ctx.device)
        total=int(group.all_reduce(x))
        assert total==sum(group.ranks)
        report[axis]=dict(local_rank=group.rank,ranks=list(group.ranks),sum=total)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dp',type=int,default=4)
    parser.add_argument('--tpsp',type=int,default=2)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--devices',help='JSON rank assignments')
    args=parser.parse_args()
    if args.dp<1 or args.tpsp<1:parser.error('mesh dimensions must be positive')
    world=World(args.dp*args.tpsp,device=args.device,
                devices=json.loads(args.devices) if args.devices else None,timeout=120)
    print(json.dumps(world.run(worker,kwargs={'dp':args.dp,'tpsp':args.tpsp}),indent=2))
