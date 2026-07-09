"""Validation / evaluation of a trained PINN.

Computes the multi-dimensional quality signals the Feedback agent scores on:
effectiveness (MSE, relative L2 against ground truth), efficiency (convergence
iterations), and robustness (loss-curve smoothness).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .interfaces import PINNProblem, TrainResult


@dataclass
class EvalReport:
    mse: float
    rel_l2: float
    loss_smoothness: float
    convergence_iters: int
    test_inputs: np.ndarray
    prediction: np.ndarray
    reference: np.ndarray | None


def _loss_smoothness(loss_history: list[float]) -> float:
    """Robustness proxy in [0, 1]: 1 = monotone/smooth descent, ->0 = spiky.

    Measured on the log-loss curve so scale doesn't dominate; the fraction of steps
    that do not increase is a cheap, stable smoothness signal.
    """

    if len(loss_history) < 2:
        return 1.0
    logs = np.log(np.clip(np.asarray(loss_history), 1e-30, None))
    diffs = np.diff(logs)
    non_increasing = np.mean(diffs <= 1e-9)
    return float(non_increasing)


def evaluate(
    problem: PINNProblem,
    result: TrainResult,
    *,
    n_test: int = 400,
    reference_inputs: Tensor | None = None,
    reference_targets: Tensor | None = None,
) -> EvalReport:
    """Evaluate a trained model against ground truth.

    Ground truth comes from ``problem.analytical`` when available, or from the
    supplied ``reference_*`` arrays (e.g. a numerically-solved / user dataset).
    """

    model = result.model
    model.eval()

    if reference_inputs is not None:
        test_x = reference_inputs
    else:
        test_x = problem.test_grid(n_test)

    with torch.no_grad():
        pred = model(test_x)

    if reference_targets is not None:
        ref = reference_targets
    elif problem.analytical is not None:
        ref = problem.analytical(test_x)
    else:
        ref = None

    if ref is not None:
        err = pred - ref
        mse = float(torch.mean(err**2))
        denom = float(torch.sqrt(torch.sum(ref**2))) or 1.0
        rel_l2 = float(torch.sqrt(torch.sum(err**2)) / denom)
    else:
        mse = float("nan")
        rel_l2 = float("nan")

    return EvalReport(
        mse=mse,
        rel_l2=rel_l2,
        loss_smoothness=_loss_smoothness(result.loss_history),
        convergence_iters=result.converged_iters,
        test_inputs=test_x.detach().cpu().numpy(),
        prediction=pred.detach().cpu().numpy(),
        reference=ref.detach().cpu().numpy() if ref is not None else None,
    )


def quality_score(report: EvalReport, weights: dict[str, float] | None = None) -> float:
    """Composite S(C) = sum w_i * m_hat_i over normalized effectiveness/eff/robustness.

    Higher is better, roughly in [0, 1]. Effectiveness dominates by default.
    """

    w = weights or {"effectiveness": 0.6, "efficiency": 0.2, "robustness": 0.2}
    effectiveness = 1.0 / (1.0 + report.rel_l2) if np.isfinite(report.rel_l2) else 0.0
    efficiency = 1.0 / (1.0 + report.convergence_iters / 1000.0)
    robustness = report.loss_smoothness
    return (
        w["effectiveness"] * effectiveness
        + w["efficiency"] * efficiency
        + w["robustness"] * robustness
    )
