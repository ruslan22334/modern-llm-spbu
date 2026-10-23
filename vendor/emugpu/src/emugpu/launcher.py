"""Launch importable user programs without depending on a lecture directory."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Callable

from .runtime import Context, World


def _resolve(entrypoint: str, project_dir: str) -> Callable:
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("entrypoint must have the form module:function")
    # Explicit user project search path, inherited identically by spawned ranks.
    # No package or demo path is inserted here.
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    target = importlib.import_module(module_name)
    for name in attribute.split("."):
        target = getattr(target, name)
    if not callable(target):
        raise TypeError(f"entrypoint {entrypoint!r} is not callable")
    return target


def _execute(ctx: Context, entrypoint: str, project_dir: str, args: tuple,
             kwargs: dict, log_dir: str | None = None) -> Any:
    print(f"[rank {ctx.rank}/{ctx.size} pid={os.getpid()} device={ctx.device}] {entrypoint}", flush=True)
    if log_dir is None:
        return _resolve(entrypoint, project_dir)(ctx, *args, **kwargs)
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"rank-{ctx.rank}.log").open("w", buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                return _resolve(entrypoint, project_dir)(ctx, *args, **kwargs)
            except BaseException:
                import traceback
                traceback.print_exc()
                raise


def launch(entrypoint: str, world_size: int = 4, device: str = "cpu", *,
           args: tuple = (), kwargs: dict | None = None,
           project_dir: str | Path | None = None, log_dir: str | Path | None = None,
           trace: str | Path | None = None, timeout: float = 30,
           debug: bool = False, debug_base_port: int = 5678,
           wait_for_debugger: str = "none", run_timeout: float | None = None, devices=None) -> list[Any]:
    """Run ``module:function(ctx, *args, **kwargs)`` in independently spawned ranks.

    Call under the main guard, as with :meth:`World.run`. ``project_dir`` defaults
    to the caller's current directory. Returns results ordered by rank. Trace and
    logs are optional explicit outputs; existing files at those paths are replaced.
    """
    if not isinstance(args, tuple) or (kwargs is not None and not isinstance(kwargs, dict)):
        raise TypeError("args must be a tuple and kwargs a dict or None")
    directory = Path(project_dir or Path.cwd()).resolve()
    if not directory.is_dir():
        raise ValueError(f"project directory does not exist: {directory}")
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("entrypoint must have the form module:function")
    logs = str(Path(log_dir).resolve()) if log_dir is not None else None
    world = World(world_size, device=device, timeout=timeout, debug=debug,
                  debug_base_port=debug_base_port, wait_for_debugger=wait_for_debugger,
                  run_timeout=run_timeout, devices=devices)
    result = world.run(_execute, args=(entrypoint, str(directory), args, kwargs or {}, logs))
    if trace is not None:
        destination = Path(trace)
        destination.parent.mkdir(parents=True, exist_ok=True)
        world.save_trace(destination)
    return result
