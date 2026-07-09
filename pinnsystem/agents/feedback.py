"""Feedback agent: score the run, then route the loop.

Scoring is deterministic (the numbers are not the LLM's to invent): metrics come from
the run's ``metrics.json`` and S(C) from the same weighted formula the evaluator uses.
The LLM — when supplied — only localizes a failure to a module and phrases the
directive; without it, a heuristic fills those in so the node is fully testable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..state import FeedbackVerdict, PINNState, QualityMetrics
from ..tools import plot_results
from .base import SupportsStructured, contract_header, invoke_structured, load_prompt

_CONTRACT = contract_header(
    "4 (Feedback)",
    prev="CodeArtifacts with a metrics.json (or a last_run_error)",
    nxt="a FeedbackVerdict routing to accept / revise_code / revert_research / await_user",
)

_WEIGHTS = {"effectiveness": 0.6, "efficiency": 0.2, "robustness": 0.2}


class _Localization(BaseModel):
    """The only judgment we defer to the LLM on a failure."""

    faulty_module: Optional[str] = None
    directive: str = ""
    architectural: bool = Field(
        False, description="True when the failure is architectural (warrants revert_research)."
    )


def _score(m: QualityMetrics) -> float:
    import math

    effectiveness = 1.0 / (1.0 + m.rel_l2) if math.isfinite(m.rel_l2) else 0.0
    efficiency = 1.0 / (1.0 + m.convergence_iters / 1000.0)
    robustness = m.loss_smoothness
    return (
        _WEIGHTS["effectiveness"] * effectiveness
        + _WEIGHTS["efficiency"] * efficiency
        + _WEIGHTS["robustness"] * robustness
    )


def _read_metrics(metrics_path: str) -> tuple[QualityMetrics, dict]:
    if not metrics_path or not Path(metrics_path).exists():
        return QualityMetrics(), {}
    try:
        raw = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return QualityMetrics(), {}
    return (
        QualityMetrics(
            mse=float(raw.get("mse", float("inf"))),
            rel_l2=float(raw.get("rel_l2", float("inf"))),
            convergence_iters=int(raw.get("convergence_iters", 0)),
            loss_smoothness=float(raw.get("loss_smoothness", 0.0)),
        ),
        raw,
    )


def _maybe_plot(raw: dict, state: PINNState) -> list[str]:
    if not all(k in raw for k in ("test_inputs", "prediction")):
        return []
    code = state.get("code")
    # Land the plot in the run's directory (metrics.json's parent), not the user's
    # dataset *file* path — spec.dataset_path is a file (often None), not a directory.
    base = Path(code.metrics_path).parent if code and code.metrics_path else Path("runs")
    out = base / "feedback_plot.png"
    try:
        result = plot_results(
            raw["test_inputs"], raw["prediction"], out, reference=raw.get("reference")
        )
        return [result["path"]]
    except (ValueError, OSError):
        return []


def feedback_node(
    state: PINNState,
    llm: Optional[SupportsStructured] = None,
) -> dict:
    """Score the latest run and decide the next transition."""

    code = state["code"]
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    threshold = state.get("accuracy_threshold", 1e-3)
    history = list(state.get("history", []))
    exhausted = iteration + 1 >= max_iter

    metrics, raw = _read_metrics(code.metrics_path if code else "")
    failed = bool(code.last_run_error) if code else True

    if failed:
        loc = _localize(state, llm)
        decision = "revert_research" if loc.architectural and not exhausted else (
            "await_user" if exhausted else "revise_code"
        )
        verdict = FeedbackVerdict(
            quality_score=0.0,
            metrics=metrics,
            passed_threshold=False,
            decision=decision,
            faulty_module=loc.faulty_module,
            directive=loc.directive or _default_directive(code),
        )
    else:
        score = _score(metrics)
        passed = metrics.rel_l2 <= threshold or metrics.mse <= threshold
        if passed:
            decision = "accept"
        elif exhausted:
            decision = "await_user"
        else:
            decision = "revise_code"
        if passed:
            directive = "Accuracy target met."
        elif decision == "await_user":
            directive = _best_so_far_directive(history, score)
        else:
            directive = (
                f"rel_l2={metrics.rel_l2:.3e} above threshold {threshold:.1e}; tighten training."
            )
        verdict = FeedbackVerdict(
            quality_score=score,
            metrics=metrics,
            passed_threshold=passed,
            decision=decision,
            directive=directive,
            plots=_maybe_plot(raw, state),
        )

    history.append(
        {
            "iteration": iteration,
            "architecture": state["research"].architecture if state.get("research") else None,
            "quality_score": verdict.quality_score,
            "rel_l2": verdict.metrics.rel_l2,
            "decision": verdict.decision,
            "forbidden": (
                state["research"].architecture
                if verdict.decision == "revert_research" and state.get("research")
                else None
            ),
        }
    )

    pending = "approve_final" if verdict.decision in {"accept", "await_user"} else None
    return {
        "feedback": verdict,
        "history": history,
        "iteration": iteration + 1,
        "pending_user_action": pending,
    }


def _localize(state: PINNState, llm: Optional[SupportsStructured]) -> _Localization:
    code = state["code"]
    if llm is None:
        return _Localization(
            faulty_module=None,
            directive=_default_directive(code),
            architectural=False,
        )
    system = f"{load_prompt('feedback')}\n\n{_CONTRACT}"
    human = (
        f"The run FAILED.\nstderr:\n{code.last_run_error}\n\n"
        f"stdout tail:\n{code.last_run_stdout[-1500:]}\n\n"
        f"Modules: {list(code.modules)}\n"
        "Attribute the failure to exactly one module and give a one-line directive. "
        "Set architectural=True only if the architecture/loss formulation itself is unworkable."
    )
    return invoke_structured(llm, _Localization, system, human)


def _best_so_far_directive(history: list[dict], current_score: float) -> str:
    """Report the best iteration so far when iterations are exhausted (rollback signal)."""

    from ..knowledge import select_best_iteration

    best = select_best_iteration(history)
    if best and best.get("quality_score", 0.0) > current_score:
        return (
            f"Iteration budget exhausted. Best result was iteration {best.get('iteration')} "
            f"(score={best['quality_score']:.3f}); consider rolling back to it."
        )
    return "Iteration budget exhausted; current result is the best so far."


def _default_directive(code) -> str:
    if code and code.last_run_error:
        first_line = code.last_run_error.strip().splitlines()[-1] if code.last_run_error.strip() else ""
        return f"Fix the module raising: {first_line}"
    return "Re-run failed with no captured error; regenerate the entrypoint."
