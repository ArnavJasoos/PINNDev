"""Pure tool logic: symbolic checks, offline-safe search, PINN ops, plotting."""

from __future__ import annotations

import numpy as np

from pinnsystem.tools import (
    build_dataset,
    evaluate_run,
    fetch_url,
    plot_results,
    symbolic_equivalence,
    sympy_parse,
    train_run,
    web_search,
)


def test_sympy_parse_ok_and_error():
    ok = sympy_parse("sin(pi*x)")
    assert ok["ok"] and "x" in ok["symbols"]

    bad = sympy_parse("this is not )( valid")
    assert not bad["ok"]
    assert bad["error"]


def test_symbolic_equivalence_symbolic_and_algebraic():
    same = symbolic_equivalence("u_xx + pi**2*sin(pi*x)", "pi**2*sin(pi*x) + u_xx")
    assert same["equivalent"]
    assert same["score"] == 1.0

    algebra = symbolic_equivalence("(x+1)**2", "x**2 + 2*x + 1")
    assert algebra["equivalent"]

    diff = symbolic_equivalence("x**2", "x**3")
    assert not diff["equivalent"]


def test_web_search_is_offline_safe():
    out = web_search("physics informed neural network", backend="none")
    assert out["results"] == []
    assert "disabled" in out["note"]


def test_fetch_url_rejects_non_web_schemes(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    out = fetch_url(secret.as_uri())  # file:// URL
    assert not out["ok"]
    assert "TOP SECRET" not in out["text"]


def test_symbolic_rejects_unsafe_tokens():
    bad = sympy_parse("().__class__.__base__.__subclasses__()")
    assert not bad["ok"]
    assert "unsafe" in bad["error"]

    eq = symbolic_equivalence("__import__('os')", "x")
    assert not eq["equivalent"]
    assert eq["method"] == "rejected"


def test_build_train_evaluate_plot_roundtrip(tmp_path):
    ds = build_dataset("poisson_1d", tmp_path / "data.npz", params={"n_points": 50})
    assert ds["n_points"] == 50

    trained = train_run(
        "poisson_1d",
        tmp_path / "out",
        hyperparams={"epochs": 200, "width": 32, "depth": 3},
        sampling={"collocation_points": 400},
    )
    assert trained["final_loss"] < 1.0

    report = evaluate_run(
        "poisson_1d",
        trained["model_path"],
        trained["config_path"],
        loss_history_path=trained["loss_history_path"],
    )
    assert np.isfinite(report["rel_l2"])
    assert 0.0 <= report["quality_score"] <= 1.0

    plotted = plot_results(
        report["test_inputs"],
        report["prediction"],
        tmp_path / "plot.png",
        reference=report["reference"],
    )
    assert plotted["backend"] in {"matplotlib", "npz-fallback"}
