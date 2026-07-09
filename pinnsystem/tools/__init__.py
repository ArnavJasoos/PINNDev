"""Pure tool implementations shared by the MCP servers and the agents.

Kept free of any ``mcp`` import so the logic is unit-testable without the server
stack, and importable directly by the skip-branch code paths. The MCP servers in
:mod:`pinnsystem.mcp` are thin wrappers over these functions.
"""

from .pinn_ops import build_dataset, evaluate_run, plot_results, train_run
from .search import arxiv_search, fetch_url, web_search
from .symbolic import symbolic_equivalence, sympy_parse

__all__ = [
    "build_dataset",
    "train_run",
    "evaluate_run",
    "plot_results",
    "web_search",
    "fetch_url",
    "arxiv_search",
    "sympy_parse",
    "symbolic_equivalence",
]
