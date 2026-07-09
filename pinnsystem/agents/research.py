"""Research agent: ProblemSpec -> self-contained ResearchReport.

Architecture selection is computed deterministically (knowledge-guided phi(E)·psi(A))
and fed to the LLM as a recommendation; the LLM fills the rest of the plan. Failure
memory from prior iterations is injected so a reverted Research never repeats itself.
"""

from __future__ import annotations

from ..knowledge import match_architecture
from ..state import PINNState, ResearchReport
from .base import SupportsStructured, contract_header, invoke_structured, load_prompt

_CONTRACT = contract_header(
    "2 (Research)",
    prev="an approved ProblemSpec with a canonical PDE + domain + feature_vector",
    nxt="a ResearchReport implementable with zero further research",
)


def _forbidden_from_history(state: PINNState) -> list[str]:
    """Collect failed architectures/approaches recorded across prior iterations."""

    forbidden: list[str] = []
    for entry in state.get("history", []):
        approach = entry.get("forbidden") or entry.get("architecture")
        if approach and approach not in forbidden:
            forbidden.append(approach)
    return forbidden


def research_node(state: PINNState, llm: SupportsStructured) -> dict:
    """Produce a `ResearchReport`, seeded with the arch match and failure memory."""

    spec = state["spec"]
    feature_vector = spec.pde.feature_vector if spec.pde else {}
    forbidden = _forbidden_from_history(state)

    match = match_architecture(feature_vector, forbidden=forbidden)

    system = f"{load_prompt('research')}\n\n{_CONTRACT}"
    human = (
        f"ProblemSpec:\n{spec.model_dump_json(indent=2)}\n\n"
        f"Architecture recommendation (phi(E).psi(A)): {match['architecture']}\n"
        f"Rationale: {match['rationale']}\n"
        f"Ranked scores: {match['scores']}\n\n"
        f"Forbidden approaches (failure memory — do NOT reuse): {forbidden or 'none'}\n\n"
        "Produce the ResearchReport."
    )

    report: ResearchReport = invoke_structured(llm, ResearchReport, system, human)

    # Deterministic guarantees the LLM must not override:
    if report.architecture in forbidden:
        report.architecture = match["architecture"]
    if not report.arch_rationale:
        report.arch_rationale = match["rationale"]
    # Carry failure memory forward so it stays visible to Coding/Feedback.
    for f in forbidden:
        if f not in report.forbidden_approaches:
            report.forbidden_approaches.append(f)

    return {"research": report}
