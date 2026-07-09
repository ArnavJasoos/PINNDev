"""Three stdio MCP servers (FastMCP) exposing the tool logic in :mod:`pinnsystem.tools`.

Servers are launchable as modules::

    python -m pinnsystem.mcp.research_server
    python -m pinnsystem.mcp.compute_server
    python -m pinnsystem.mcp.pinn_server

:mod:`pinnsystem.mcp.client` wires them into a LangChain ``MultiServerMCPClient``.
"""
