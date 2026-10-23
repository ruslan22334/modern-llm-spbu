"""Adjoints of generic collectives with different losses on each mesh rank."""
import unittest
import torch
from emugpu import World
from emugpu import collectives


def worker(ctx):
    replicas=2 if ctx.size==4 else 1
    mesh=ctx.mesh((replicas,ctx.size//replicas),('replica','model'))
    comm=mesh['model'].comm
    p,r=comm.size,comm.rank
    coefficient=r+1
    total_coeff=p*(p+1)/2
    x=torch.full((2,),float(r+1),dtype=torch.float64,requires_grad=True)
    gathered=collectives.all_gather(x,comm)
    (gathered.sum()*coefficient).backward()
    torch.testing.assert_close(x.grad,torch.full_like(x,total_coeff))
    x=torch.full((2,),float(r+1),dtype=torch.float64,requires_grad=True)
    reduced=collectives.all_reduce(x,comm)
    (reduced.square().sum()*coefficient).backward()
    torch.testing.assert_close(x.grad,torch.full_like(x,2*total_coeff**2))
    x=(torch.arange(2*p,dtype=torch.float64)+r).requires_grad_()
    local=collectives.reduce_scatter(x,comm)
    (local.square().sum()*coefficient).backward()
    full_sum=p*torch.arange(2*p,dtype=torch.float64)+p*(p-1)/2
    coefficients=torch.arange(1,p+1,dtype=torch.float64).repeat_interleave(2)
    torch.testing.assert_close(x.grad,2*full_sum*coefficients)
    return True


class CollectiveAdjointsTests(unittest.TestCase):
    def test_independent_losses_and_mesh_groups(self):
        for size in (1,2,4): self.assertEqual(World(size).run(worker),[True]*size)

if __name__=='__main__':unittest.main()
