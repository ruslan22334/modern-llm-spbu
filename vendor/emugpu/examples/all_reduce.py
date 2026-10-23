import torch
from emugpu import World


def worker(ctx):
    value=torch.tensor([ctx.rank+1.],device=ctx.device)
    return dict(rank=ctx.rank,total=float(ctx.all_reduce(value)))

if __name__=='__main__': print(World(4).run(worker))
