"""Parser agent: natural language -> canonical, user-approvable ProblemSpec."""

from __future__ import annotations

from ..state import PINNState, ProblemSpec
from .base import SupportsStructured, contract_header, invoke_structured, load_prompt

_CONTRACT = contract_header(
    "1 (Parser)",
    prev="the raw user request only",
    nxt="a ProblemSpec with a canonical SymbolicPDE, DomainSpec, quantities, and data flags",
)


def parser_node(state: PINNState, llm: SupportsStructured) -> dict:
    """Produce a `ProblemSpec` and flag whether user approval is still pending."""

    spec_in = state["spec"]
    raw_query = spec_in.raw_query

    system = f"{load_prompt('parser')}\n\n{_CONTRACT}"
    human = (
        f"User request:\n{raw_query}\n\n"
        "Produce the ProblemSpec. Enumerate every assumption in normalized_statement "
        "and set approved_by_user=False unless the request is already unambiguous and "
        "explicitly complete."
    )

    spec: ProblemSpec = invoke_structured(llm, ProblemSpec, system, human)

    # The raw query is authoritative; never let the model rewrite it.
    spec.raw_query = raw_query

    # The statement is always confirmed by a human before the loop starts, regardless
    # of how confident the model was — the human_clarify gate flips this to True.
    spec.approved_by_user = False
    return {"spec": spec, "pending_user_action": "approve_statement"}
