import os
import unittest
import torch
from emugpu import World, EmulationError


def worker(ctx, permuted=False):
    ranks = [3,1,2,0] if permuted else None
    mesh = ctx.mesh((2,2),("rows","cols"),ranks=ranks)
    for name in ("rows","cols"):
        c = mesh[name].get_group()
        x = torch.tensor([[float(ctx.global_rank)]])
        torch.testing.assert_close(c.all_gather(x),torch.tensor(c.ranks,dtype=torch.float32).reshape(-1,1))
        torch.testing.assert_close(c.all_reduce(x),torch.tensor([[float(sum(c.ranks))]]))
        full = torch.full((2,1),float(ctx.global_rank))
        torch.testing.assert_close(c.reduce_scatter(full),torch.tensor([[float(sum(c.ranks))]]))
        b = c.broadcast(torch.tensor([9]) if c.rank==1 else None,src=1)
        assert int(b)==9
        incoming = c.all_to_all([torch.tensor([[100*ctx.global_rank+c.ranks[d]]]) for d in range(2)])
        assert [int(b) for b in incoming]==[100*src+ctx.global_rank for src in c.ranks]
        received = c.send_recv(x,dst=(c.rank+1)%2,src=(c.rank-1)%2,tag=20)
        assert int(received)==c.ranks[(c.rank-1)%2]
        receive = c.irecv((c.rank-1)%2,tag=21)
        send = c.isend(x,(c.rank+1)%2,tag=21)
        assert int(receive.wait())==c.ranks[(c.rank-1)%2]
        send.wait(); send.wait()
        gathered = c.gather(x,dst=1)
        assert (gathered is not None)==(c.rank==1)
        reduced = c.reduce(x,dst=1)
        assert (reduced is not None)==(c.rank==1)
        scat = c.scatter(torch.arange(4) if c.rank==1 else None,src=1)
        torch.testing.assert_close(scat,torch.arange(4).reshape(2,2)[c.rank])
        uneven = torch.full((c.rank,1),ctx.global_rank,dtype=torch.int64)
        gathered_v = c.all_gather_v(uneven)
        assert [b.shape[0] for b in gathered_v]==[0,1]
        pending = c.isend(torch.tensor([ctx.global_rank]),(c.rank+1)%2,tag=57)
        a2a = c.all_to_all_single(torch.tensor([[c.rank*10],[c.rank*10+1]]))
        torch.testing.assert_close(a2a,torch.tensor([[c.rank],[10+c.rank]]))
        receipt = c.irecv((c.rank-1)%2,tag=57)
        assert int(receipt.wait())==c.ranks[(c.rank-1)%2]
        pending.wait()
        trace=c.gather_trace()
        if c.rank==0: assert trace and all(e['group'] in ([*mesh['rows'].comm.ranks],[*mesh['cols'].comm.ranks]) for e in trace if e['op']!='new_group')
    both = mesh['rows','cols'].get_group()
    assert both.size==4
    assert int(both.all_reduce(torch.tensor([1])))==4
    return dict(rank=ctx.rank,pid=os.getpid(),coordinate=mesh.get_coordinate())


def any_size(ctx):
    mesh=ctx.mesh((ctx.size,),("cols",))
    c=mesh['cols'].comm
    assert int(c.all_reduce(torch.tensor([1])))==ctx.size
    assert int(c.send_recv(torch.tensor([c.rank]),(c.rank+1)%c.size,(c.rank-1)%c.size,tag=40))==(c.rank-1)%c.size
    return ctx.rank


def contract_errors(ctx):
    for shape,names,ranks in ((None,('cols',),None),((2,),('cols','cols'),None),
                              ((ctx.size,),('cols',),[0.0,1.0])):
        try:
            ctx.mesh(shape if ctx.rank==0 else (ctx.size,),
                     names if ctx.rank==0 else ('cols',), ranks=ranks if ctx.rank==0 else None)
        except EmulationError: pass
        else: raise AssertionError('invalid/asymmetric mesh input accepted')
        ctx.barrier()
    mesh=ctx.mesh((ctx.size,),('cols',))
    group=mesh['cols'].comm
    for op in ('reduce','gather'):
        try: getattr(group,op)(torch.ones(1),dst=100 if group.rank==0 else 0)
        except EmulationError: pass
        else: raise AssertionError('invalid destination accepted')
        group.barrier()
    return True


class MeshTests(unittest.TestCase):
    def test_groups_order_and_primitives(self):
        for permuted in (False,True):
            reports=World(4,timeout=60).run(worker,args=(permuted,))
            self.assertEqual(len({r['pid'] for r in reports}),4)

    def test_contract_errors_leave_group_usable(self):
        self.assertEqual(World(2).run(contract_errors),[True,True])

    def test_single_rank(self):
        self.assertEqual(World(1).run(any_size),[0])

    def test_unlimited_validation(self):
        self.assertEqual(World(16).size,16)
        for n in (0,-1,True):
            with self.assertRaises(ValueError): World(n)
        with self.assertRaises(ValueError): World(4,devices=['cpu'])

if __name__=='__main__': unittest.main()
