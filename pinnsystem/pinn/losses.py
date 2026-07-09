"""Autograd differential operators and the composite PINN loss.

The PDE residual is problem-specific (supplied via ``PINNProblem.residual_fn``), so
this module stays generic: it provides the derivative helpers residual functions are
written in terms of, plus the weighted assembly of residual + boundary/initial terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .interfaces import PINNProblem


def grad(outputs: Tensor, inputs: Tensor) -> Tensor:
    """d(outputs)/d(inputs), retaining the graph for higher-order derivatives."""

    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def partial(outputs: Tensor, inputs: Tensor, index: int) -> Tensor:
    """Partial derivative of a scalar output w.r.t. a single input column."""

    g = grad(outputs, inputs)
    return g[:, index : index + 1]


def laplacian(outputs: Tensor, inputs: Tensor) -> Tensor:
    """Sum of unmixed second derivatives over all input dimensions."""

    g = grad(outputs, inputs)
    lap = torch.zeros_like(outputs)
    for i in range(inputs.shape[1]):
        gi = g[:, i : i + 1]
        lap = lap + grad(gi, inputs)[:, i : i + 1]
    return lap


@dataclass
class LossComponents:
    total: Tensor
    residual: Tensor
    boundary: Tensor


def pinn_loss(
    model: nn.Module,
    problem: PINNProblem,
    collocation: Tensor,
    *,
    residual_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> LossComponents:
    """Weighted sum of the interior PDE residual and the boundary/initial residual."""

    coords = collocation.clone().requires_grad_(True)
    res = problem.residual_fn(model, coords)
    residual_loss = torch.mean(res**2)

    bc_res = problem.boundary_fn(model)
    boundary_loss = torch.mean(bc_res**2)

    total = residual_weight * residual_loss + boundary_weight * boundary_loss
    return LossComponents(total=total, residual=residual_loss, boundary=boundary_loss)
