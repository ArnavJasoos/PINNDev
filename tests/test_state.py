"""State schema validation and config/llm_factory smoke tests."""

import pytest

from pinnsystem.config import AppConfig, llm_factory, load_config
from pinnsystem.state import (
    HyperParams,
    PINNState,
    ProblemSpec,
    ResearchReport,
    new_state,
)


def test_new_state_shape():
    st = new_state("solve a 1D heat equation on [0,1]", accuracy_threshold=1e-4, max_iterations=5)
    assert st["spec"].raw_query.startswith("solve")
    assert st["accuracy_threshold"] == 1e-4
    assert st["max_iterations"] == 5
    assert st["iteration"] == 0
    assert st["research"] is None


def test_problem_spec_defaults():
    spec = ProblemSpec(raw_query="q")
    assert spec.approved_by_user is False
    assert spec.pde is None


def test_research_report_defaults():
    rr = ResearchReport()
    assert rr.architecture == "MLP"
    assert isinstance(rr.hyperparams, HyperParams)


def test_config_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.provider in {"anthropic", "openai", "ollama"}
    assert cfg.resolved_model()


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("PINN_PROVIDER", "openai")
    monkeypatch.setenv("PINN_MODEL", "gpt-4o-mini")
    cfg = load_config()
    assert cfg.provider == "openai"
    assert cfg.resolved_model() == "gpt-4o-mini"


def test_llm_factory_unknown_provider():
    with pytest.raises(ValueError):
        llm_factory(provider="does-not-exist")  # type: ignore[arg-type]
