"""Export standalone trace viewer and per-rank debugger configuration."""
from __future__ import annotations

from importlib import resources
import json
from pathlib import Path


def _write(path: str | Path, content: str, *, overwrite: bool) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(content)
    return destination


def export_viewer(path: str | Path = "trace_viewer.html", *, overwrite: bool = False) -> Path:
    """Copy the self-contained HTML viewer; open it and select a trace JSON."""
    content = resources.files("emugpu.assets").joinpath("trace_viewer.html").read_text(encoding="utf-8")
    return _write(path, content, overwrite=overwrite)


def export_debug_config(path: str | Path = ".vscode/launch.json", world_size: int = 4,
                        base_port: int = 5678, *, overwrite: bool = False) -> Path:
    """Write VS Code/debugpy attach sessions plus an attach-all compound."""
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if isinstance(base_port, bool) or not isinstance(base_port, int) or not 1 <= base_port <= 65536 - world_size:
        raise ValueError("base_port plus world_size must fit TCP ports 1–65535")
    sessions = [{
        "name": f"emugpu rank {rank}", "type": "debugpy", "request": "attach",
        "connect": {"host": "127.0.0.1", "port": base_port + rank},
        "justMyCode": True,
    } for rank in range(world_size)]
    config = {"version": "0.2.0", "configurations": sessions,
              "compounds": [{"name": f"emugpu: attach all {world_size} ranks",
                             "configurations": [session["name"] for session in sessions],
                             "stopAll": True}]}
    return _write(path, json.dumps(config, indent=2) + "\n", overwrite=overwrite)
