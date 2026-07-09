"""Agent node functions, driven by a fake structured LLM (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from pinnsystem.agents import coding_node, feedback_node, parser_node, research_node
from pinnsystem.agents.coding import GeneratedCode
from pinnsystem.execution import new_workspace
from pinnsystem.knowledge import match_architecture
from pinnsystem.state import (
    CodeArtifacts,
    DomainSpec,
    ProblemSpec,
    ResearchReport,
    SymbolicPDE,
    new_state,
)


class FakeLLM:
    """Returns a canned Pydantic instance for any structured call."""

    def __init__(self, response):
        self._response = response

    def with_structured_output(self, schema):  # noqa: ARG002 - schema ignored intentionally
        return _Bound(self._response)


class _Bound:
    def __init__(self, response):
        self._response = response

    def invoke(self, _messages):
        return self._response


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parser_preserves_query_and_flags_approval():
    state = new_state("solve the 1D heat equation on [0,1]")
    llm = FakeLLM(
        ProblemSpec(
            raw_query="MODEL SHOULD NOT OVERWRITE THIS",
            normalized_statement="u_t = alpha u_xx on [0,1]",
            approved_by_user=False,
        )
    )
    out = parser_node(state, llm)
    assert out["spec"].raw_query == "solve the 1D heat equation on [0,1]"
    assert out["pending_user_action"] == "approve_statement"


# --------------------------------------------------------------------------- #
# Research
# --------------------------------------------------------------------------- #


def test_research_overrides_forbidden_architecture():
    spec = ProblemSpec(
        raw_query="oscillatory multi-scale problem",
        approved_by_user=True,
        pde=SymbolicPDE(
            feature_vector={"periodicity": 0.9, "geometry_complexity": 0.2, "multiscale": 0.9}
        ),
        domain=DomainSpec(dims=1, variables=["x"]),
    )
    state = new_state("oscillatory")
    state["spec"] = spec
    state["history"] = [{"forbidden": "MLP"}]

    llm = FakeLLM(ResearchReport(architecture="MLP"))  # forbidden — must be replaced
    out = research_node(state, llm)
    report = out["research"]

    expected = match_architecture(spec.pde.feature_vector, forbidden=["MLP"])["architecture"]
    assert report.architecture != "MLP"
    assert report.architecture == expected
    assert "MLP" in report.forbidden_approaches
    assert report.arch_rationale


# --------------------------------------------------------------------------- #
# Coding
# --------------------------------------------------------------------------- #


def test_coding_runs_entrypoint_and_collects_metrics(tmp_path):
    ws = new_workspace(tmp_path, run_id="coderun")
    good_main = (
        "import json\n"
        "json.dump({'mse':1e-4,'rel_l2':2e-3,'convergence_iters':120,'loss_smoothness':0.9},"
        " open('metrics.json','w'))\n"
        "print('run ok')\n"
    )
    llm = FakeLLM(GeneratedCode(modules={"main.py": good_main}, entrypoint="main.py"))

    state = new_state("x")
    state["research"] = ResearchReport()
    out = coding_node(state, llm, workspace=ws, code_debug_budget=2, run_timeout=30)

    code = out["code"]
    assert code.last_run_error is None
    assert Path(code.metrics_path).name == "metrics.json"
    assert "run ok" in code.last_run_stdout


def test_coding_self_debug_exhausts_budget(tmp_path):
    ws = new_workspace(tmp_path, run_id="failrun")
    bad_main = "raise RuntimeError('module boom')\n"
    llm = FakeLLM(GeneratedCode(modules={"main.py": bad_main}, entrypoint="main.py"))

    state = new_state("x")
    state["research"] = ResearchReport()
    out = coding_node(state, llm, workspace=ws, code_debug_budget=2, run_timeout=30)

    assert out["code"].last_run_error is not None
    assert "boom" in out["code"].last_run_error


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #


def _state_with_metrics(tmp_path, metrics: dict, error=None):
    mpath = tmp_path / "metrics.json"
    mpath.write_text(json.dumps(metrics), encoding="utf-8")
    state = new_state("x", accuracy_threshold=1e-2, max_iterations=3)
    state["research"] = ResearchReport(architecture="MLP")
    state["code"] = CodeArtifacts(metrics_path=str(mpath), last_run_error=error)
    return state


def test_feedback_accepts_when_below_threshold(tmp_path):
    state = _state_with_metrics(
        tmp_path, {"mse": 1e-4, "rel_l2": 1e-3, "convergence_iters": 100, "loss_smoothness": 0.95}
    )
    out = feedback_node(state)  # no LLM needed
    v = out["feedback"]
    assert v.decision == "accept"
    assert v.passed_threshold
    assert 0.0 < v.quality_score <= 1.0
    assert out["pending_user_action"] == "approve_final"
    assert len(out["history"]) == 1


def test_feedback_revises_when_above_threshold(tmp_path):
    state = _state_with_metrics(
        tmp_path, {"mse": 1.0, "rel_l2": 0.5, "convergence_iters": 5000, "loss_smoothness": 0.4}
    )
    out = feedback_node(state)
    assert out["feedback"].decision == "revise_code"
    assert not out["feedback"].passed_threshold


def test_feedback_localizes_failure_without_llm(tmp_path):
    state = _state_with_metrics(tmp_path, {}, error="Traceback...\nValueError: bad shape in pde_loss")
    out = feedback_node(state)
    v = out["feedback"]
    assert v.decision == "revise_code"
    assert "pde_loss" in v.directive
