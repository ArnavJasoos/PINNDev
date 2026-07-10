"""Assemble the PINN StateGraph.

langgraph is imported lazily inside :func:`build_graph` so importing this module (for
the node wrappers or routing) doesn't require the ``agents`` extra. The graph wires the
four agents plus two human-in-the-loop interrupt nodes, with the conditional edges
defined in :mod:`pinnsystem.graph.routing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..agents import (
    coding_node,
    feedback_node,
    intent_router_node,
    parser_node,
    research_node,
)
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

    def intent_router(state: PINNState) -> dict:
        return intent_router_node(state, deps.llm)

    return {
        "parser": parser,
        "research": research,
        "coding": coding,
        "feedback": feedback,
        "intent_router": intent_router,
    }


def _make_human_nodes() -> dict[str, Any]:
    from langgraph.types import interrupt

    def human_clarify(state: PINNState) -> dict:
        spec = state["spec"]
        pde = spec.pde
        domain = spec.domain
        payload = interrupt(
            {
                "type": "clarify",
                "statement": spec.normalized_statement,
                "pde_latex": pde.latex if pde else "",
                "operators": list(pde.operators) if pde else [],
                "boundary_conditions": list(pde.boundary_conditions) if pde else [],
                "initial_conditions": list(pde.initial_conditions) if pde else [],
                "domain": (
                    {
                        "dims": domain.dims,
                        "variables": list(domain.variables),
                        "bounds": {k: list(v) for k, v in domain.bounds.items()},
                    }
                    if domain
                    else None
                ),
                "quantities": list(spec.quantities),
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

    # After the user confirms the problem statement (human_clarify), the research→
    # coding→feedback loop runs to completion by default: an "accept" verdict goes
    # straight to END instead of stopping at a second human gate. Set
    # extra["require_user_approval"]=True to re-enable the final approval dialog.
    require_approval = deps.config.extra.get("require_user_approval", False)

    # Entry: follow-up → intent router, else data-dependency branching.
    graph.add_conditional_edges(
        START,
        routing.entry_route,
        {
            routing.PARSER: "parser",
            routing.RESEARCH: "research",
            routing.INTENT_ROUTER: "intent_router",
        },
    )

    # Intent router → the stage that owns the mid-session change.
    graph.add_conditional_edges(
        "intent_router",
        routing.intent_route,
        {routing.PARSER: "parser", routing.RESEARCH: "research", routing.CODING: "coding"},
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


def _state_serde():
    """Serializer that trusts our own ``pinnsystem.state`` Pydantic models.

    LangGraph's msgpack layer warns (and will soon *block*) when it deserializes a
    checkpoint holding a type outside its default allowlist. Our state models are all
    that end up in the checkpoint, so we register them explicitly — this silences the
    "Deserializing unregistered type ..." warnings and keeps resume working once strict
    mode becomes the default. Derived from the module so new models stay covered.
    """

    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError:  # pragma: no cover - older langgraph without the allowlist API
        return None

    from pydantic import BaseModel

    from .. import state as _state

    allow = [
        (_state.__name__, name)
        for name, obj in vars(_state).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel)
    ]
    try:
        return JsonPlusSerializer(allowed_msgpack_modules=allow)
    except TypeError:  # pragma: no cover - kwarg absent on this version
        return None


def _make_checkpointer(checkpoint_path: Optional[str]):
    """SQLite checkpointer (persistent) or in-memory fallback."""

    serde = _state_serde()
    saver_kwargs = {"serde": serde} if serde is not None else {}

    if checkpoint_path:
        try:
            from pathlib import Path

            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            # The GUI drives the graph with astream(), so the checkpointer must
            # support async ops. AsyncSqliteSaver lazily connects on first use.
            conn = aiosqlite.connect(checkpoint_path)
            return AsyncSqliteSaver(conn, **saver_kwargs)
        except ImportError:  # pragma: no cover - optional sqlite extra
            pass
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(**saver_kwargs)
