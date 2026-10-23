"""Route full matrices to owners and back; no optimizer algorithm included."""
import argparse
import json
import torch
from emugpu import World


def worker(ctx):
    group=ctx.mesh((ctx.size,),('owners',))['owners'].comm
    # Mock gradient matrices of different sizes. Shapes are a shared manifest,
    # as they would be known from named parameters of the same model.
    shapes=[(rank+2,3) for rank in range(group.size)]
    gradient=torch.arange((group.rank+2)*3,dtype=torch.float64,device=ctx.device).reshape(shapes[group.rank])
    owner=(group.rank+1)%group.size
    blocks=[gradient.flatten() if destination==owner else gradient.new_empty(0)
            for destination in range(group.size)]
    received=group.all_to_all(blocks)
    returned=[]
    for source,flat in enumerate(received):
        if flat.numel():
            matrix=flat.reshape(shapes[source])
            # Insert your optimizer's local matrix transformation here.
            # The example only clones the matrix to verify round-trip routing.
            returned.append(matrix.clone().flatten())
        else:returned.append(flat)
    result=group.all_to_all(returned)[owner].reshape(shapes[group.rank])
    torch.testing.assert_close(result,gradient)
    # Unequal output/update chunks from one owner can also be distributed:
    chunks=[torch.full((rank,2),rank,dtype=torch.float64,device=ctx.device)
            for rank in range(group.size)] if group.rank==0 else None
    local=group.scatter_v(chunks,src=0)
    gathered=group.gather_v(local,dst=0)
    if group.rank==0:
        for rank,chunk in enumerate(gathered):torch.testing.assert_close(chunk,chunks[rank])
    return dict(rank=ctx.global_rank,owner=owner,shape=list(gradient.shape),status='PASS')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--world-size',type=int,default=4)
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    print(json.dumps(World(args.world_size,device=args.device,timeout=120).run(worker),indent=2))
