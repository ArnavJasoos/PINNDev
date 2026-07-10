"""End-to-end control-flow proof: the compiled graph runs to END after the user
confirms the problem statement — no LLM/network, no torch training.

This is the Phase 1 regression for the "only the parser ran" bug: it drives the real
StateGraph (parser -> clarify interrupt -> research -> coding -> feedback -> END) with a
schema-dispatching fake LLM and a trivial entrypoint, so a wiring break or a swallowed
exception anywhere in the loop fails the test instead of silently stopping.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from pinnsystem.agents.coding import GeneratedCode
from pinnsystem.agents.router import RouteDecision
from pinnsystem.config import AppConfig
from pinnsystem.execution import VenvRunner, new_workspace
from pinnsystem.graph.builder import GraphDeps, build_graph
from pinnsystem.state import DomainSpec, ProblemSpec, ResearchReport, SymbolicPDE, new_state

# A metrics.json below the accuracy threshold so Feedback decides "accept".
_GOOD_MAIN = (
    "import json\n"
    "json.dump({'mse':1e-5,'rel_l2':2e-4,'convergence_iters':100,'loss_smoothness':0.95},"
    " open('metrics.json','w'))\n"
    "print('run ok')\n"
)


class _SchemaDispatchLLM:
    """Fake structured LLM returning a canned instance keyed by the requested schema."""

    def __init__(self, responses: dict) -> None:
        self._responses = responses

    def with_structured_output(self, schema):
        try:
            response = self._responses[schema]
        except KeyError:  # pragma: no cover - guards a mis-wired test
            raise AssertionError(f"no canned response for schema {schema!r}")
        return _Bound(response)


class _Bound:
    def __init__(self, response) -> None:
        self._response = response

    def invoke(self, _messages):
        return self._response


def _build(tmp_path, *, route_target: str = "research"):
    llm = _SchemaDispatchLLM(
        {
            ProblemSpec: ProblemSpec(
                raw_query="ignored",
                normalized_statement="u_tt + u = 0 on [0, 5]",
                pde=SymbolicPDE(latex="u_{tt} + u = 0", operators=["d2/dt2"]),
                domain=DomainSpec(dims=1, variables=["t"], bounds={"t": (0.0, 5.0)}),
                quantities=["u"],
                approved_by_user=True,  # parser_node must force this back to False
            ),
            ResearchReport: ResearchReport(architecture="MLP", arch_rationale="baseline"),
            GeneratedCode: GeneratedCode(modules={"main.py": _GOOD_MAIN}, entrypoint="main.py"),
            RouteDecision: RouteDecision(target=route_target, reason="test"),
        }
    )
    workspace = new_workspace(tmp_path, run_id="e2e")
    deps = GraphDeps(llm=llm, workspace=workspace, config=AppConfig(), runner=VenvRunner())
    graph = build_graph(deps, checkpoint_path=None)  # MemorySaver: sync invoke is fine
    return graph, workspace


def test_graph_completes_after_statement_confirmation(tmp_path):
    graph, _ = _build(tmp_path)
    config = {"configurable": {"thread_id": "e2e"}}

    init = new_state("solve the undamped oscillator", accuracy_threshold=1e-3, max_iterations=3)

    # 1. Parser runs, then the graph halts at the statement-confirmation interrupt.
    first = graph.invoke(init, config)
    assert "__interrupt__" in first, "expected a clarify interrupt after the parser"

    # The confirmation payload carries the parsed problem for the user to review.
    data = first["__interrupt__"][0].value
    assert data["type"] == "clarify"
    assert data["pde_latex"] == "u_{tt} + u = 0"
    assert data["domain"]["variables"] == ["t"]
    assert data["quantities"] == ["u"]

    # 2. User approves the statement -> the loop must run all the way to END.
    from langgraph.types import Command

    final = graph.invoke(Command(resume={"approved": True}), config)

    assert "__interrupt__" not in final, "loop stopped early instead of running to completion"
    verdict = final["feedback"]
    assert verdict is not None, "feedback never ran — loop did not reach the end"
    assert verdict.decision == "accept"
    assert verdict.passed_threshold


def test_statement_correction_reparses_before_looping(tmp_path):
    graph, _ = _build(tmp_path)
    config = {"configurable": {"thread_id": "correct"}}

    first = graph.invoke(new_state("solve the undamped oscillator"), config)
    assert "__interrupt__" in first

    # A correction (approved=False) must route back to the parser, not into research.
    from langgraph.types import Command

    second = graph.invoke(
        Command(resume={"approved": False, "answer": "it's damped, add friction"}), config
    )
    assert "__interrupt__" in second, "correction should re-run the parser and re-confirm"
    assert second["research"] is None, "research must not start before the statement is approved"

    final = graph.invoke(Command(resume={"approved": True}), config)
    assert "__interrupt__" not in final
    assert final["feedback"].decision == "accept"


def test_followup_smart_reroutes_without_reparsing(tmp_path):
    # route_target="research": a follow-up should re-enter at research, skipping the
    # parser and its confirmation gate entirely, and run straight through to END.
    graph, _ = _build(tmp_path, route_target="research")
    config = {"configurable": {"thread_id": "followup"}}

    from langgraph.types import Command

    graph.invoke(new_state("solve the undamped oscillator", max_iterations=5), config)
    done = graph.invoke(Command(resume={"approved": True}), config)
    assert done["feedback"].decision == "accept"

    # Now a mid-session change on the same thread — no new statement confirmation.
    after = graph.invoke({"followup": "make the network deeper"}, config)
    assert "__interrupt__" not in after, "follow-up should not stop at a clarify gate"
    assert after["followup_target"] == "research"
    assert after["revision_note"] is None, "coding must clear the note after honoring it"
    assert after["feedback"].decision == "accept"


def test_followup_targeting_problem_reconfirms(tmp_path):
    # route_target="parser": a follow-up that changes the problem must re-run the parser
    # and pause at the confirmation gate again.
    graph, _ = _build(tmp_path, route_target="parser")
    config = {"configurable": {"thread_id": "reconfirm"}}

    from langgraph.types import Command

    graph.invoke(new_state("solve the undamped oscillator", max_iterations=5), config)
    graph.invoke(Command(resume={"approved": True}), config)

    after = graph.invoke({"followup": "actually it's a 2D heat equation"}, config)
    assert "__interrupt__" in after, "a problem change must re-confirm the statement"
