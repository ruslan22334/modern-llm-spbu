"""Named Cartesian process meshes. All construction is collective on WORLD."""
from __future__ import annotations
import itertools
import math
import torch
from .runtime import Context, EmulationError


class MeshView:
    """A flattened selection of axes, fixing all other coordinates locally."""
    def __init__(self, parent, names, comm):
        self.parent, self.mesh_dim_names, self.comm = parent, names, comm
        self.shape = tuple(parent.shape[parent.mesh_dim_names.index(n)] for n in names)

    def get_group(self):
        return self.comm

    def get_local_rank(self):
        return self.comm.rank

    def size(self):
        return self.comm.size

    def get_coordinate(self):
        return [self.parent.coordinate[self.parent.mesh_dim_names.index(n)] for n in self.mesh_dim_names]


class DeviceMesh:
    """Torch-like named mesh; ``mesh['tp'].get_group()`` returns Context.

    Each global rank occurs exactly once. Mesh order determines logical rank
    order, even for a permuted layout. Default row-major shape (2, 2) is [[0,1],
    [2,3]]. All ranks construct meshes and request combined axes in the same
    order. Single-axis groups are built eagerly. This object is not a torch
    DeviceMesh / DTensor process-group argument: use the Context API.
    """
    def __init__(self, ctx: Context, shape, names, *, ranks=None):
        error = None
        layout = None
        try:
            shape, names = tuple(shape), tuple(names)
            if not shape or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape):
                raise ValueError("mesh shape must contain positive integers")
            if len(names) != len(shape) or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
                raise ValueError("mesh needs a unique nonempty name for each axis")
            if math.prod(shape) != ctx.size:
                raise ValueError("product of mesh dimensions must equal world size")
            layout = list(range(ctx.size)) if ranks is None else torch.as_tensor(ranks).flatten().tolist()
            if any(isinstance(r, bool) or not isinstance(r, int) for r in layout) or sorted(layout) != list(range(ctx.size)):
                raise ValueError("mesh ranks must be an integer permutation of all global world ranks")
        except (TypeError, ValueError, RuntimeError) as exc:
            error = str(exc)
        ctx._preflight("mesh", local_error=error, shape=shape if error is None else (),
                       names=names if error is None else (), ranks=layout if error is None else ())
        if ctx.group is not torch.distributed.group.WORLD:
            raise EmulationError("build meshes from a world context")
        self.ctx, self.shape, self.mesh_dim_names = ctx, shape, names
        self.mesh = torch.tensor(layout, dtype=torch.int64).reshape(shape)
        self.coordinate = tuple(int(v) for v in (self.mesh == ctx.global_rank).nonzero()[0])
        self._views = {}
        for name in names:
            self._build((name,))

    def _build(self, names):
        indices = tuple(self.mesh_dim_names.index(n) for n in names)
        fixed = tuple(i for i in range(len(self.shape)) if i not in indices)
        own = None
        for fixed_values in itertools.product(*(range(self.shape[i]) for i in fixed)):
            members = []
            for varying in itertools.product(*(range(self.shape[i]) for i in indices)):
                coordinate = [0] * len(self.shape)
                for i, value in zip(fixed, fixed_values): coordinate[i] = value
                for i, value in zip(indices, varying): coordinate[i] = value
                members.append(int(self.mesh[tuple(coordinate)]))
            group = self.ctx.new_group(members)
            if group is not None: own = group
        view = MeshView(self, names, own)
        self._views[names] = view
        return view

    def __getitem__(self, names):
        names = (names,) if isinstance(names, str) else tuple(names)
        if not names or len(set(names)) != len(names) or any(n not in self.mesh_dim_names for n in names):
            raise KeyError(f"unknown/duplicate mesh axes: {names}")
        # Combined axes follow the caller's order, as in a tensor permutation.
        return self._views[names] if names in self._views else self._build(names)

    def get_group(self, mesh_dim=None):
        if mesh_dim is None:
            if len(self.shape) != 1: raise ValueError("specify an axis for a multidimensional mesh")
            mesh_dim = self.mesh_dim_names[0]
        if isinstance(mesh_dim, int): mesh_dim = self.mesh_dim_names[mesh_dim]
        return self[mesh_dim].get_group()

    group = get_group

    def get_coordinate(self):
        return list(self.coordinate)

    def size(self, mesh_dim=None):
        return math.prod(self.shape) if mesh_dim is None else self.get_group(mesh_dim).size

    def get_local_rank(self, mesh_dim=None):
        return self.get_group(mesh_dim).rank


def init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None, ctx=None, ranks=None):
    """Build after initialize(), or pass the worker's existing world Context."""
    ctx = Context(device_type) if ctx is None else ctx
    names = tuple(f"dim_{i}" for i in range(len(mesh_shape))) if mesh_dim_names is None else mesh_dim_names
    return DeviceMesh(ctx, mesh_shape, names, ranks=ranks)
