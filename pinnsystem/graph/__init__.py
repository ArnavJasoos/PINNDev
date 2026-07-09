"""LangGraph orchestration: routing logic (pure) + StateGraph assembly (lazy).

The conditional-edge decisions live in :mod:`pinnsystem.graph.routing` as plain
functions so the routing table is unit-testable without langgraph installed. The
StateGraph itself is built in :mod:`pinnsystem.graph.builder`, which imports langgraph
lazily so the rest of the package stays importable on a core-only install.
"""

from .routing import (
    after_clarify,
    after_final_approval,
    entry_route,
    feedback_route,
)

__all__ = [
    "entry_route",
    "after_clarify",
    "feedback_route",
    "after_final_approval",
]
