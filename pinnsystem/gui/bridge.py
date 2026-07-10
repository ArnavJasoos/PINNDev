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


def event_to_transcript(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Turn one ``astream`` node update into a transcript entry.

    Returns ``{stage, label, detail, files, body}``: ``detail`` is the collapsed
    one-liner, ``files`` the artifacts the stage produced, and ``body`` the expandable
    verbose text (hyperparams, stdout tail, metrics) for the "Thinking" panel.
    """

    label = _STAGE_LABELS.get(node, node)
    detail = ""
    files: list[str] = []
    body = ""

    if node == "parser" and update.get("spec"):
        detail = update["spec"].normalized_statement
    elif node == "research" and update.get("research"):
        r = update["research"]
        detail = f"{r.architecture}: {r.arch_rationale}"
        hp = r.hyperparams
        body = (
            f"Architecture: {r.architecture}\n"
            f"Hyperparams: width={hp.width} depth={hp.depth} lr={hp.lr} "
            f"epochs={hp.epochs} opt={hp.optimizer} act={hp.activation}\n"
            f"Loss terms: {', '.join(f'{t.name}(w={t.weight})' for t in r.loss_terms) or '—'}\n"
            f"Sampling: {r.sampling.collocation_points} collocation / "
            f"{r.sampling.boundary_points} boundary / {r.sampling.initial_points} initial"
        )
    elif node == "coding" and update.get("code"):
        c = update["code"]
        detail = "run ok" if not c.last_run_error else f"error: {c.last_run_error.splitlines()[-1]}"
        files = [p for p in c.modules.values()]
        for extra in (c.dataset_path, c.model_path, c.metrics_path):
            if extra:
                files.append(extra)
        body = (c.last_run_stdout or "")[-2000:]
        if c.last_run_error:
            body = f"{body}\n\n--- stderr ---\n{c.last_run_error}"[-2000:]
    elif node == "feedback" and update.get("feedback"):
        v = update["feedback"]
        detail = f"decision={v.decision} score={v.quality_score:.3f} — {v.directive}"
        m = v.metrics
        body = (
            f"mse={m.mse:.3e} rel_l2={m.rel_l2:.3e} "
            f"convergence_iters={m.convergence_iters} loss_smoothness={m.loss_smoothness:.3f}\n"
            f"passed_threshold={v.passed_threshold}"
        )
        files = list(v.plots)

    return {"stage": node, "label": label, "detail": detail, "files": files, "body": body}


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
