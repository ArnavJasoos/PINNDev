"""Glue between the LangGraph run and the NiceGUI UI.

The pure helpers here — initial-state construction and event→transcript mapping — are
independent of nicegui/langgraph so they can be unit-tested. :class:`PinnRunner` drives
the compiled graph's ``astream`` and threads human-in-the-loop resumes; it imports the
graph builder lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from ..state import PINNState, ProblemSpec, new_state

# Human-readable one-liners per stage for the transcript panel.
_STAGE_LABELS = {
    "parser": "Parser — canonical problem statement",
    "research": "Research — architecture & training plan",
    "coding": "Coding — generated modules & run",
    "feedback": "Feedback — scored the run",
    "human_clarify": "Awaiting your clarification/approval",
    "human_approve_final": "Awaiting your final approval",
}


def initial_state_from_input(
    query: str,
    *,
    dataset_path: Optional[str] = None,
    formulas_given: bool = False,
    accuracy_threshold: float = 1e-3,
    max_iterations: int = 3,
) -> PINNState:
    """Build the starting :class:`PINNState`, setting the data-dependency flags."""

    state = new_state(query, accuracy_threshold=accuracy_threshold, max_iterations=max_iterations)
    spec: ProblemSpec = state["spec"]
    spec.user_provided_dataset = dataset_path is not None
    spec.dataset_path = dataset_path
    spec.user_provided_formulas = formulas_given
    return state


def event_to_transcript(node: str, update: dict[str, Any]) -> dict[str, str]:
    """Turn one ``astream`` node update into a transcript entry {stage, label, detail}."""

    label = _STAGE_LABELS.get(node, node)
    detail = ""
    if node == "parser" and update.get("spec"):
        detail = update["spec"].normalized_statement
    elif node == "research" and update.get("research"):
        r = update["research"]
        detail = f"{r.architecture}: {r.arch_rationale}"
    elif node == "coding" and update.get("code"):
        c = update["code"]
        detail = "run ok" if not c.last_run_error else f"error: {c.last_run_error.splitlines()[-1]}"
    elif node == "feedback" and update.get("feedback"):
        v = update["feedback"]
        detail = f"decision={v.decision} score={v.quality_score:.3f} — {v.directive}"
    return {"stage": node, "label": label, "detail": detail}


@dataclass
class PinnRunner:
    """Drives a compiled graph for one GUI session (one ``thread_id``)."""

    graph: Any
    thread_id: str = "gui-session"
    transcript: list[dict] = field(default_factory=list)

    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    async def stream(self, state_or_command: Any) -> AsyncIterator[dict]:
        """Stream node updates, appending each to the transcript.

        Yields either a transcript entry ``{stage,...}`` or an interrupt marker
        ``{"interrupt": payload}`` the UI turns into a dialog.
        """

        async for chunk in self.graph.astream(state_or_command, self._config()):
            if "__interrupt__" in chunk:
                yield {"interrupt": chunk["__interrupt__"]}
                continue
            for node, update in chunk.items():
                entry = event_to_transcript(node, update or {})
                self.transcript.append(entry)
                yield entry

    def resume_command(self, payload: dict) -> Any:
        """Build the ``Command(resume=...)`` used to answer an interrupt."""

        from langgraph.types import Command

        return Command(resume=payload)
