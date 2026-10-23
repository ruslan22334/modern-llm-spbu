import unittest
import torch
from emugpu import World, EmulationError
from emugpu.runtime import normalize_devices


def worker(ctx):
    root=ctx.size-1
    mesh=ctx.mesh((1,ctx.size),('row','col'),ranks=list(reversed(range(ctx.size))))
    c=mesh['col'].comm
    chunks=[torch.full((rank,3),rank,dtype=torch.float64) for rank in range(c.size)] if c.rank==root else None
    value=c.scatter_v(chunks,src=root)
    torch.testing.assert_close(value,torch.full((c.rank,3),c.rank,dtype=torch.float64))
    values=c.gather_v(value,dst=root)
    if c.rank==root:
        for r,b in enumerate(values):torch.testing.assert_close(b,torch.full((r,3),r,dtype=torch.float64))
    else:assert values is None
    splits=list(range(c.size))
    full=torch.arange(sum(splits),dtype=torch.float64)+c.rank
    reduced=c.reduce_scatter_v(full,splits,op='mean')
    expected=(torch.arange(sum(splits),dtype=torch.float64)+(c.size-1)/2).split(splits)[c.rank]
    torch.testing.assert_close(reduced,expected)
    # Ragged 1D exchange supports flattening matrices of different shapes.
    splits=[(c.rank+destination)%3 for destination in range(c.size)]
    blocks=[torch.full((n,),10*c.rank+d,dtype=torch.float64) for d,n in enumerate(splits)]
    incoming=c.all_to_all_single(torch.cat(blocks),input_split_sizes=splits,
                                 output_split_sizes=[(source+c.rank)%3 for source in range(c.size)])
    expected=torch.cat([torch.full(((source+c.rank)%3,),10*source+c.rank,dtype=torch.float64) for source in range(c.size)])
    torch.testing.assert_close(incoming,expected)
    if c.size>1:
        try:c.scatter_v([torch.ones(1)] if c.rank==root else None,src=root)
        except EmulationError:c.barrier()
        else:raise AssertionError('invalid source block count accepted')
    return True


def hybrid(ctx,dp,tpsp):
    mesh=ctx.mesh((dp,tpsp),('dp','tpsp'))
    data,model=mesh['dp'].comm,mesh['tpsp'].comm
    d,t=mesh.get_coordinate()
    assert model.ranks==tuple(range(d*tpsp,(d+1)*tpsp))
    assert data.ranks==tuple(replica*tpsp+t for replica in range(dp))
    x=torch.tensor([ctx.global_rank])
    assert int(data.all_reduce(x))==sum(data.ranks)
    assert int(model.all_reduce(x))==sum(model.ranks)
    return ctx.global_rank


class ExchangeTests(unittest.TestCase):
    def test_ragged_gather_scatter_and_flat_a2a(self):
        for n in (1,4):self.assertEqual(World(n).run(worker),[True]*n)

    def test_dp4_model2_groups(self):
        self.assertEqual(World(8,timeout=60).run(hybrid,args=(4,2)),list(range(8)))

    def test_dp8_model2_over_eight_processes(self):
        self.assertEqual(World(16,timeout=90).run(hybrid,args=(8,2)),list(range(16)))

    def test_multi_device_assignment_validation(self):
        assignments=normalize_devices(4,devices=[[f'cuda:{2*r}',f'cuda:{2*r+1}'] for r in range(4)])
        self.assertEqual([[str(d) for d in group] for group in assignments],
                         [['cuda:0','cuda:1'],['cuda:2','cuda:3'],['cuda:4','cuda:5'],['cuda:6','cuda:7']])
        self.assertEqual(World(32).size,32)
        self.assertEqual(len(World(4,devices=[['cpu','cpu']]*4).rank_devices[0]),2)

if __name__=='__main__':unittest.main()
