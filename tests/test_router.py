"""Intent router: classify a follow-up and stage it for the right node."""

from __future__ import annotations

from pinnsystem.agents.router import RouteDecision, intent_router_node
from pinnsystem.state import ProblemSpec, ResearchReport, new_state


class _FakeLLM:
    def __init__(self, decision):
        self._decision = decision

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return self._decision


def _state(followup: str):
    st = new_state("solve the oscillator")
    st["spec"] = ProblemSpec(raw_query="solve the oscillator", normalized_statement="u_tt+u=0")
    st["research"] = ResearchReport(architecture="MLP")
    st["followup"] = followup
    return st


def test_router_stages_note_and_clears_followup():
    st = _state("make it deeper")
    out = intent_router_node(st, _FakeLLM(RouteDecision(target="research")))
    assert out["followup_target"] == "research"
    assert out["revision_note"] == "make it deeper"
    assert out["followup"] is None
    assert "spec" not in out  # research route leaves the statement untouched


def test_router_parser_target_reopens_statement():
    st = _state("actually it's 2D")
    out = intent_router_node(st, _FakeLLM(RouteDecision(target="parser")))
    assert out["followup_target"] == "parser"
    assert out["spec"].approved_by_user is False
    assert "[follow-up] actually it's 2D" in out["spec"].raw_query


def test_router_empty_followup_is_safe():
    st = _state("")
    out = intent_router_node(st, _FakeLLM(RouteDecision(target="coding")))
    assert out["followup"] is None
    assert out["revision_note"] is None
