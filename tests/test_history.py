"""Outcome history cache H + best-iteration rollback helper."""

from __future__ import annotations

from pinnsystem.knowledge import HistoryStore, select_best_iteration


def test_history_store_records_and_returns_best(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    store.record("poisson_1d", "MLP", 0.7, rel_l2=0.1, passed=False)
    store.record("poisson_1d", "Fourier-MLP", 0.95, rel_l2=1e-3, passed=True)
    store.record("heat_2d", "SIREN", 0.8)

    best = store.best_architecture("poisson_1d")
    assert best["architecture"] == "Fourier-MLP"
    assert best["passed"]

    assert store.best_architecture("unknown") is None


def test_select_best_iteration():
    history = [
        {"iteration": 0, "quality_score": 0.4},
        {"iteration": 1, "quality_score": 0.8},
        {"iteration": 2, "quality_score": 0.6},
    ]
    best = select_best_iteration(history)
    assert best["iteration"] == 1

    assert select_best_iteration([]) is None
    assert select_best_iteration([{"iteration": 0}]) is None  # no score
