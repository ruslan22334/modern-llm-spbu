import torch
from emugpu import World


def worker(ctx):
    previous=(ctx.rank-1)%ctx.size; next_rank=(ctx.rank+1)%ctx.size
    value=torch.tensor([ctx.rank],device=ctx.device)
    receive=ctx.irecv(previous,tag=7)
    send=ctx.isend(value,next_rank,tag=7)
    incoming=receive.wait();send.wait()
    return dict(rank=ctx.rank,received=int(incoming))

if __name__=='__main__': print(World(4).run(worker))
