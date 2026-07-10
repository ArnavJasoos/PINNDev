"""Intent router: classify a mid-session follow-up into the stage it should re-enter.

When a user tweaks a completed run ("make the network deeper", "the domain should be
[0, 10]", "the data loader is wrong"), the follow-up should not restart the whole
pipeline. This node asks the LLM which stage owns the change and re-enters there,
reusing the checkpointed state:

* ``parser``   — the problem itself changed (re-parse and re-confirm the statement).
* ``research`` — the architecture / hyperparameters / plan should change.
* ``coding``   — the generated code is wrong but the plan is fine.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..state import PINNState
from .base import SupportsStructured, invoke_structured

_SYSTEM = (
    "You route a user's follow-up request to the pipeline stage that owns the change. "
    "Choose 'parser' if the physics/problem statement, domain, PDE, or quantities change "
    "(this re-confirms the statement with the user). Choose 'research' if the architecture, "
    "loss weighting, sampling, or hyperparameters should change. Choose 'coding' if the "
    "plan is fine but the generated code is buggy or should be implemented differently."
)


class RouteDecision(BaseModel):
    """The router's single decision."""

    target: Literal["parser", "research", "coding"] = "research"
    reason: str = Field("", description="One line explaining the choice.")


def intent_router_node(state: PINNState, llm: SupportsStructured) -> dict:
    """Classify ``state['followup']`` and stage it for the chosen node."""

    followup = (state.get("followup") or "").strip()
    if not followup:
        # Nothing to route (defensive): fall through to research without a note.
        return {"followup": None, "followup_target": "research", "revision_note": None}

    human = (
        f"Follow-up request:\n{followup}\n\n"
        f"Current statement: {state['spec'].normalized_statement or state['spec'].raw_query}\n"
        f"Current architecture: "
        f"{state['research'].architecture if state.get('research') else '(none yet)'}\n"
        "Return the stage that should handle this."
    )
    decision: RouteDecision = invoke_structured(llm, RouteDecision, _SYSTEM, human)

    updates: dict = {
        "followup": None,
        "followup_target": decision.target,
        "revision_note": followup,
    }
    if decision.target == "parser":
        # Fold the change into the raw query and force re-confirmation of the statement.
        spec = state["spec"]
        spec.raw_query = f"{spec.raw_query}\n\n[follow-up] {followup}"
        spec.approved_by_user = False
        updates["spec"] = spec
    return updates
