"""PyTorch process launcher, named meshes, communication primitives and manual DP.

Gloo CPU transport is used for both CPU and explicitly assigned CUDA tensors.
"""
from .runtime import CommunicationTimeout, Context, EmulationError, World, Work, configure_debugger, initialize, run
from .launcher import launch
from .mesh import DeviceMesh, MeshView, init_device_mesh
from .resources import export_debug_config, export_viewer
from .dp import DataParallel, broadcast_module, sync_gradients

__version__ = "0.3.2"
__all__ = ["CommunicationTimeout", "Context", "EmulationError", "World", "Work",
           "configure_debugger", "initialize", "run", "launch", "DeviceMesh", "MeshView",
           "init_device_mesh", "export_debug_config", "export_viewer", "DataParallel",
           "broadcast_module", "sync_gradients", "__version__"]
