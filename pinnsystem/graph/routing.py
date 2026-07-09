"""Pure conditional-edge decisions for the PINN StateGraph.

Each function maps a state to the *name* of the next node (or a sentinel). Keeping
them free of langgraph means the whole routing table can be tested with plain dicts,
and the graph builder just references them.
"""

from __future__ import annotations

from ..state import PINNState

# Node name constants (shared by routing + builder so they never drift).
PARSER = "parser"
RESEARCH = "research"
CODING = "coding"
FEEDBACK = "feedback"
HUMAN_CLARIFY = "human_clarify"
HUMAN_APPROVE_FINAL = "human_approve_final"
END = "__end__"


def entry_route(state: PINNState) -> str:
    """Data-dependency branching at graph entry.

    Flags are set on the initial ``ProblemSpec`` (e.g. by the GUI) before the run:
    a dataset skips data generation; given formulas skip physics parsing/research.
    """

    spec = state["spec"]
    if spec.user_provided_formulas:
        # Formulas (and maybe data) given → straight to train-only Research.
        return RESEARCH
    if spec.user_provided_dataset:
        # Data but no formulas → still parse the problem (lightly) against the data.
        return PARSER
    return PARSER


def after_clarify(state: PINNState) -> str:
    """After a clarification round: proceed only once the user approved the statement."""

    spec = state["spec"]
    if spec.approved_by_user:
        return RESEARCH
    return PARSER


def feedback_route(state: PINNState, *, require_user_approval: bool = True) -> str:
    """Route on the Feedback verdict, honoring the iteration cap and dual exit."""

    verdict = state.get("feedback")
    if verdict is None:
        return HUMAN_APPROVE_FINAL

    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    if iteration >= max_iter:
        # Cap reached — report best-so-far to the user regardless of decision.
        return HUMAN_APPROVE_FINAL

    decision = verdict.decision
    if decision == "accept":
        return HUMAN_APPROVE_FINAL if require_user_approval else END
    if decision == "revise_code":
        return CODING
    if decision == "revert_research":
        return RESEARCH
    # await_user or anything unexpected → hand to the user.
    return HUMAN_APPROVE_FINAL


def after_final_approval(state: PINNState) -> str:
    """After the final human gate: END if approved, else loop back to Research."""

    spec = state["spec"]
    if spec.approved_by_user:
        return END
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        # No iterations left to improve — end even without explicit approval.
        return END
    return RESEARCH
