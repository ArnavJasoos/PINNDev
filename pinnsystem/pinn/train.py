"""Standardized PINN training loop.

Consumes a :class:`PINNProblem` plus the Research agent's ``HyperParams`` /
``SamplingPlan`` and returns a :class:`TrainResult`. Supports Adam and (optionally)
an L-BFGS refinement pass — the two optimizers PINNs almost always use.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..state import HyperParams, SamplingPlan
from .architectures import build_network
from .interfaces import PINNProblem, TrainResult
from .losses import pinn_loss


def train_pinn(
    problem: PINNProblem,
    hyperparams: Optional[HyperParams] = None,
    sampling: Optional[SamplingPlan] = None,
    *,
    residual_weight: float = 1.0,
    boundary_weight: float = 1.0,
    model: Optional[nn.Module] = None,
    seed: int = 0,
    log_every: int = 0,
    convergence_tol: float = 1e-5,
) -> TrainResult:
    """Train a PINN on ``problem`` and report loss history + convergence.

    ``convergence_tol`` records the first epoch where total loss drops below it
    (0 if never), feeding the Feedback agent's efficiency metric.
    """

    hp = hyperparams or HyperParams()
    sp = sampling or SamplingPlan()

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    if model is None:
        model = build_network(
            "MLP",
            problem.input_dim,
            problem.output_dim,
            width=hp.width,
            depth=hp.depth,
            activation=hp.activation,
            seed=seed,
        )

    collocation = problem.sample_interior(sp.collocation_points, generator=gen)

    result = TrainResult(model=model)

    def closure_loss() -> torch.Tensor:
        comps = pinn_loss(
            model,
            problem,
            collocation,
            residual_weight=residual_weight,
            boundary_weight=boundary_weight,
        )
        result._last = comps  # type: ignore[attr-defined]
        return comps.total

    def record(comps) -> None:
        result.loss_history.append(float(comps.total.detach()))
        result.residual_history.append(float(comps.residual.detach()))
        result.boundary_history.append(float(comps.boundary.detach()))
        if result.converged_iters == 0 and comps.total.detach() < convergence_tol:
            result.converged_iters = len(result.loss_history)

    if hp.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=hp.lr, max_iter=20, line_search_fn="strong_wolfe"
        )
        for epoch in range(hp.epochs):
            def closure():
                optimizer.zero_grad()
                loss = closure_loss()
                loss.backward()
                return loss

            optimizer.step(closure)
            comps = result._last  # type: ignore[attr-defined]
            record(comps)
            if log_every and epoch % log_every == 0:
                print(f"[{problem.name}] lbfgs epoch {epoch} loss={comps.total.item():.3e}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        for epoch in range(hp.epochs):
            optimizer.zero_grad()
            comps = pinn_loss(
                model,
                problem,
                collocation,
                residual_weight=residual_weight,
                boundary_weight=boundary_weight,
            )
            comps.total.backward()
            optimizer.step()
            record(comps)
            if log_every and epoch % log_every == 0:
                print(f"[{problem.name}] adam epoch {epoch} loss={comps.total.item():.3e}")

    result.final_loss = result.loss_history[-1] if result.loss_history else float("inf")
    if result.converged_iters == 0:
        result.converged_iters = len(result.loss_history)
    return result
