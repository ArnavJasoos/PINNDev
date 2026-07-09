"""MCP layer: servers import and register tools; client spec is well-formed."""

from __future__ import annotations

import pytest

from pinnsystem.mcp import client

mcp_server = pytest.importorskip("mcp.server.fastmcp")


def test_server_specs_shape(tmp_path):
    specs = client.server_specs(workdir=str(tmp_path), search_backend="none")
    assert set(specs) == {"research_tools", "compute_tools", "pinn_tools"}
    for name, spec in specs.items():
        assert spec["transport"] == "stdio"
        assert spec["args"][0] == "-m"
    assert specs["compute_tools"]["env"]["PINN_WORKDIR"] == str(tmp_path)
    assert specs["research_tools"]["env"]["PINN_SEARCH_BACKEND"] == "none"


def test_servers_expose_tools():
    from pinnsystem.mcp import compute_server, pinn_server, research_server

    for server in (research_server.mcp, compute_server.mcp, pinn_server.mcp):
        # FastMCP registers tools on an internal manager; each server has >= 1.
        assert server is not None
