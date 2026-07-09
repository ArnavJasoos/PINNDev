"""End-to-end PINN core proof: a hand-written 1D problem trains to threshold.

De-risks the hardest technical piece (PINN training actually converges) before any
LLM/agent is wired in — this is Build Order phase 2's exit criterion.
"""

import pytest

torch = pytest.importorskip("torch")

from pinnsystem.pinn.evaluate import evaluate, quality_score
from pinnsystem.pinn.problems import damped_oscillator, poisson_1d
from pinnsystem.pinn.train import train_pinn
from pinnsystem.state import HyperParams, SamplingPlan


def test_poisson_1d_trains_to_threshold():
    problem = poisson_1d()
    hp = HyperParams(width=32, depth=3, lr=5e-3, epochs=3000, activation="tanh")
    sp = SamplingPlan(collocation_points=256)

    result = train_pinn(problem, hp, sp, boundary_weight=10.0, seed=0)
    report = evaluate(problem, result, n_test=200)

    assert report.rel_l2 < 1e-2, f"rel_l2={report.rel_l2}"
    # composite S weights efficiency/robustness in; a well-converged solve sits ~0.8
    assert quality_score(report) > 0.75


def test_damped_oscillator_trains_reasonably():
    problem = damped_oscillator(omega=2.0, zeta=0.1, t_max=5.0)
    hp = HyperParams(width=64, depth=4, lr=5e-3, epochs=4000, activation="tanh")
    sp = SamplingPlan(collocation_points=512)

    result = train_pinn(problem, hp, sp, boundary_weight=20.0, seed=0)
    report = evaluate(problem, result, n_test=200)

    # oscillatory ODE is harder for a plain MLP; loose bound proves convergence works
    assert report.rel_l2 < 2e-1, f"rel_l2={report.rel_l2}"
