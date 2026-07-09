"""Pure routing table for the StateGraph (no langgraph needed)."""

from __future__ import annotations

from pinnsystem.graph import routing
from pinnsystem.state import FeedbackVerdict, ProblemSpec, new_state


def _state(**over):
    st = new_state("solve heat eq", max_iterations=3)
    st.update(over)
    return st


def test_entry_route_branches_on_flags():
    st = _state()
    st["spec"] = ProblemSpec(raw_query="x", user_provided_formulas=True)
    assert routing.entry_route(st) == routing.RESEARCH

    st["spec"] = ProblemSpec(raw_query="x", user_provided_dataset=True)
    assert routing.entry_route(st) == routing.PARSER

    st["spec"] = ProblemSpec(raw_query="x")
    assert routing.entry_route(st) == routing.PARSER


def test_after_clarify_gates_on_approval():
    st = _state()
    st["spec"] = ProblemSpec(raw_query="x", approved_by_user=False)
    assert routing.after_clarify(st) == routing.PARSER
    st["spec"] = ProblemSpec(raw_query="x", approved_by_user=True)
    assert routing.after_clarify(st) == routing.RESEARCH


def test_feedback_route_decisions():
    st = _state(iteration=1)
    st["feedback"] = FeedbackVerdict(decision="revise_code")
    assert routing.feedback_route(st) == routing.CODING

    st["feedback"] = FeedbackVerdict(decision="revert_research")
    assert routing.feedback_route(st) == routing.RESEARCH

    st["feedback"] = FeedbackVerdict(decision="accept")
    assert routing.feedback_route(st, require_user_approval=True) == routing.HUMAN_APPROVE_FINAL
    assert routing.feedback_route(st, require_user_approval=False) == routing.END


def test_feedback_route_respects_iteration_cap():
    st = _state(iteration=3)  # >= max_iterations
    st["feedback"] = FeedbackVerdict(decision="revise_code")
    assert routing.feedback_route(st) == routing.HUMAN_APPROVE_FINAL


def test_after_final_approval():
    st = _state(iteration=0)
    st["spec"] = ProblemSpec(raw_query="x", approved_by_user=True)
    assert routing.after_final_approval(st) == routing.END

    st["spec"] = ProblemSpec(raw_query="x", approved_by_user=False)
    assert routing.after_final_approval(st) == routing.RESEARCH

    st["iteration"] = 3  # exhausted
    assert routing.after_final_approval(st) == routing.END
