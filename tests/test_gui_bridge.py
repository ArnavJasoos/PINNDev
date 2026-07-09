"""Framework-free GUI glue: initial state + event→transcript mapping."""

from __future__ import annotations

from pinnsystem.gui.bridge import event_to_transcript, initial_state_from_input
from pinnsystem.state import CodeArtifacts, FeedbackVerdict, ResearchReport


def test_initial_state_sets_data_flags():
    st = initial_state_from_input("solve heat eq", dataset_path="d.npz", formulas_given=True)
    spec = st["spec"]
    assert spec.user_provided_dataset and spec.dataset_path == "d.npz"
    assert spec.user_provided_formulas
    assert st["spec"].raw_query == "solve heat eq"


def test_initial_state_no_dataset():
    st = initial_state_from_input("q", max_iterations=5)
    assert not st["spec"].user_provided_dataset
    assert st["max_iterations"] == 5


def test_event_to_transcript_research_and_feedback():
    r = event_to_transcript("research", {"research": ResearchReport(architecture="SIREN", arch_rationale="osc")})
    assert r["stage"] == "research"
    assert "SIREN" in r["detail"]

    f = event_to_transcript(
        "feedback", {"feedback": FeedbackVerdict(decision="accept", quality_score=0.9)}
    )
    assert "accept" in f["detail"]


def test_event_to_transcript_coding_error():
    c = event_to_transcript(
        "coding", {"code": CodeArtifacts(last_run_error="Traceback\nValueError: bad")}
    )
    assert "error" in c["detail"]
    assert "ValueError: bad" in c["detail"]
