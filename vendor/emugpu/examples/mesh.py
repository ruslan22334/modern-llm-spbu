import torch
from emugpu import World


def worker(ctx):
    # DP, TPSP
    mesh=ctx.mesh((2,2),('rows','cols'))
    result=dict(rank=ctx.global_rank,coordinate=mesh.get_coordinate())
    for axis in mesh.mesh_dim_names: 
        comm=mesh[axis].get_group() # rows
        value=torch.tensor([ctx.global_rank],device=ctx.device)
        # [0], [1]
        # [2], [3]
        # 0.  1   1. 1
        # 2.  5   3. 5

        # 0.  2   1. 4
        # 2.  2   3. 4
        result[axis]=dict(ranks=comm.ranks,total=int(comm.all_reduce(value)))
    return result

if __name__=='__main__': print(World(4).run(worker))

#
