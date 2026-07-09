"""MCP server: compute tools (run_python in an isolated subprocess, file I/O).

The workdir is fixed per server process via ``PINN_WORKDIR`` (defaults to CWD); all
paths are resolved *inside* it and path traversal outside is refused, so a generated
script can only touch its own run directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..execution import VenvRunner

mcp = FastMCP("compute_tools")

_runner = VenvRunner(os.environ.get("PINN_VENV_PYTHON") or None)


def _workroot() -> Path:
    return Path(os.environ.get("PINN_WORKDIR", ".")).resolve()


def _safe_path(relative: str) -> Path:
    root = _workroot()
    candidate = (root / relative).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"Path {relative!r} escapes the run workdir.")
    return candidate


@mcp.tool()
def run_python(code: str, timeout: float = 300.0, filename: str = "_snippet.py") -> dict[str, Any]:
    """Run a Python snippet in the isolated interpreter; capture stdout/stderr."""

    # Strip any directory components so `filename` cannot escape the run workdir
    # (Path joins discard the left side on an absolute right side).
    safe_name = Path(filename).name or "_snippet.py"
    outcome = _runner.run_code(code, workdir=_workroot(), filename=safe_name, timeout=timeout)
    return {
        "ok": outcome.ok,
        "returncode": outcome.returncode,
        "stdout": outcome.stdout,
        "stderr": outcome.stderr,
        "timed_out": outcome.timed_out,
        "error": outcome.error,
    }


@mcp.tool()
def write_file(relative_path: str, content: str) -> dict[str, Any]:
    """Write ``content`` to a file inside the run workdir."""

    path = _safe_path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "bytes": len(content.encode("utf-8"))}


@mcp.tool()
def read_file(relative_path: str, max_chars: int = 50000) -> dict[str, Any]:
    """Read a file from inside the run workdir."""

    path = _safe_path(relative_path)
    if not path.exists():
        return {"ok": False, "text": "", "error": "file not found"}
    return {"ok": True, "text": path.read_text(encoding="utf-8")[:max_chars], "error": None}


@mcp.tool()
def list_workdir(relative_path: str = ".") -> dict[str, Any]:
    """List entries in a directory inside the run workdir."""

    path = _safe_path(relative_path)
    if not path.exists():
        return {"entries": [], "error": "not found"}
    entries = [
        {"name": p.name, "is_dir": p.is_dir(), "size": p.stat().st_size if p.is_file() else 0}
        for p in sorted(path.iterdir())
    ]
    return {"entries": entries, "error": None}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
