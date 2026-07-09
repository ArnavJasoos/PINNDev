"""MCP server: research tools (web/arxiv search, fetch, sympy parse + equivalence)."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..tools import arxiv_search, fetch_url, symbolic_equivalence, sympy_parse, web_search

mcp = FastMCP("research_tools")


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web for a query. Backend/key come from PINN_SEARCH_* env vars."""

    return web_search(
        query,
        backend=os.environ.get("PINN_SEARCH_BACKEND", "duckduckgo"),
        api_key=os.environ.get("PINN_SEARCH_API_KEY"),
        max_results=max_results,
    )


@mcp.tool()
def fetch(url: str, max_chars: int = 20000) -> dict[str, Any]:
    """Fetch the text content of a URL (truncated to ``max_chars``)."""

    return fetch_url(url, max_chars=max_chars)


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search arXiv for relevant papers."""

    return arxiv_search(query, max_results=max_results)


@mcp.tool()
def parse_expression(expr_src: str) -> dict[str, Any]:
    """Parse a sympy expression, returning LaTeX, symbols, and canonical form."""

    return sympy_parse(expr_src)


@mcp.tool()
def check_equivalence(lhs: str, rhs: str) -> dict[str, Any]:
    """Composite symbolic+numeric equivalence check between two residual expressions."""

    return symbolic_equivalence(lhs, rhs)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
