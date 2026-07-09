"""Assemble the PINN StateGraph.

langgraph is imported lazily inside :func:`build_graph` so importing this module (for
the node wrappers or routing) doesn't require the ``agents`` extra. The graph wires the
four agents plus two human-in-the-loop interrupt nodes, with the conditional edges
defined in :mod:`pinnsystem.graph.routing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..agents import coding_node, feedback_node, parser_node, research_node
from ..agents.base import SupportsStructured
from ..config import AppConfig
from ..execution import RunWorkspace, VenvRunner
from ..state import PINNState
from . import routing


@dataclass
class GraphDeps:
    """Everything the node wrappers need, bound once at build time."""

    llm: SupportsStructured
    workspace: RunWorkspace
    config: AppConfig
    runner: Optional[VenvRunner] = None


# --------------------------------------------------------------------------- #
# Node wrappers — adapt agent functions to the (state) -> partial signature.
# --------------------------------------------------------------------------- #


def _make_nodes(deps: GraphDeps) -> dict[str, Any]:
    runner = deps.runner or VenvRunner()

    def parser(state: PINNState) -> dict:
        return parser_node(state, deps.llm)

    def research(state: PINNState) -> dict:
        return research_node(state, deps.llm)

    def coding(state: PINNState) -> dict:
        return coding_node(
            state,
            deps.llm,
            workspace=deps.workspace,
            runner=runner,
            code_debug_budget=deps.config.code_debug_budget,
        )

    def feedback(state: PINNState) -> dict:
        return feedback_node(state, deps.llm)

    return {"parser": parser, "research": research, "coding": coding, "feedback": feedback}


def _make_human_nodes() -> dict[str, Any]:
    from langgraph.types import interrupt

    def human_clarify(state: PINNState) -> dict:
        spec = state["spec"]
        payload = interrupt(
            {
                "type": "clarify",
                "statement": spec.normalized_statement,
                "question": state.get("pending_user_action"),
            }
        )
        # payload is whatever the GUI resumes with: {approved: bool, answer: str}
        if payload.get("approved"):
            spec.approved_by_user = True
        elif payload.get("answer"):
            spec.raw_query = f"{spec.raw_query}\n\n[clarification] {payload['answer']}"
        return {"spec": spec, "pending_user_action": None}

    def human_approve_final(state: PINNState) -> dict:
        spec = state["spec"]
        payload = interrupt(
            {
                "type": "approve_final",
                "feedback": state.get("feedback").model_dump() if state.get("feedback") else None,
            }
        )
        spec.approved_by_user = bool(payload.get("approved"))
        return {"spec": spec, "pending_user_action": None}

    return {"human_clarify": human_clarify, "human_approve_final": human_approve_final}


def build_graph(deps: GraphDeps, *, checkpoint_path: Optional[str] = None):
    """Compile the StateGraph with a SQLite checkpointer for interrupt/resume."""

    from langgraph.graph import END, START, StateGraph

    nodes = _make_nodes(deps)
    humans = _make_human_nodes()

    graph = StateGraph(PINNState)
    for name, fn in {**nodes, **humans}.items():
        graph.add_node(name, fn)

    require_approval = deps.config.extra.get("require_user_approval", True)

    # Entry: data-dependency branching.
    graph.add_conditional_edges(
        START,
        routing.entry_route,
        {routing.PARSER: "parser", routing.RESEARCH: "research"},
    )

    # Parser → clarify interrupt → (loop | research).
    graph.add_edge("parser", "human_clarify")
    graph.add_conditional_edges(
        "human_clarify",
        routing.after_clarify,
        {routing.PARSER: "parser", routing.RESEARCH: "research"},
    )

    # Research → Coding → Feedback.
    graph.add_edge("research", "coding")
    graph.add_edge("coding", "feedback")

    # Feedback verdict routing.
    graph.add_conditional_edges(
        "feedback",
        lambda s: routing.feedback_route(s, require_user_approval=require_approval),
        {
            routing.CODING: "coding",
            routing.RESEARCH: "research",
            routing.HUMAN_APPROVE_FINAL: "human_approve_final",
            routing.END: END,
        },
    )

    # Final human gate.
    graph.add_conditional_edges(
        "human_approve_final",
        routing.after_final_approval,
        {routing.RESEARCH: "research", routing.END: END},
    )

    checkpointer = _make_checkpointer(checkpoint_path)
    return graph.compile(checkpointer=checkpointer)


def _make_checkpointer(checkpoint_path: Optional[str]):
    """SQLite checkpointer (persistent) or in-memory fallback."""

    if checkpoint_path:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            return SqliteSaver.from_conn_string(checkpoint_path)
        except ImportError:  # pragma: no cover - optional sqlite extra
            pass
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()
