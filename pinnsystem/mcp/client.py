"""LangChain MultiServerMCPClient wiring for the three PINN MCP servers.

Produces the stdio server spec (which module to launch, env to pass) and, when
``langchain-mcp-adapters`` is installed, builds a ready ``MultiServerMCPClient``.
Kept import-light: the adapter package is imported lazily so this module can be
imported (and the spec inspected/tested) without the agent extra installed.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional


def server_specs(
    *,
    workdir: str,
    venv_python: Optional[str] = None,
    search_backend: str = "duckduckgo",
    search_api_key: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Build the stdio ``MultiServerMCPClient`` config for all three servers."""

    compute_env = {"PINN_WORKDIR": workdir}
    if venv_python:
        compute_env["PINN_VENV_PYTHON"] = venv_python

    research_env = {"PINN_SEARCH_BACKEND": search_backend}
    if search_api_key:
        research_env["PINN_SEARCH_API_KEY"] = search_api_key

    def spec(module: str, env: dict[str, str]) -> dict[str, Any]:
        return {
            "command": sys.executable,
            "args": ["-m", module],
            "transport": "stdio",
            "env": {**os.environ, **env},
        }

    return {
        "research_tools": spec("pinnsystem.mcp.research_server", research_env),
        "compute_tools": spec("pinnsystem.mcp.compute_server", compute_env),
        "pinn_tools": spec("pinnsystem.mcp.pinn_server", {}),
    }


def build_client(
    *,
    workdir: str,
    venv_python: Optional[str] = None,
    search_backend: str = "duckduckgo",
    search_api_key: Optional[str] = None,
):
    """Return a ``MultiServerMCPClient`` for the three servers.

    Raises ImportError with an actionable message if the agent extra is missing.
    """

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "MCP client needs `pip install .[agents]` (langchain-mcp-adapters)."
        ) from exc

    specs = server_specs(
        workdir=workdir,
        venv_python=venv_python,
        search_backend=search_backend,
        search_api_key=search_api_key,
    )
    return MultiServerMCPClient(specs)
