"""Reference PINN problems used to prove the core trains end-to-end.

These hand-written problems de-risk the numerics before any LLM is wired in, and they
double as concrete few-shot examples the Coding agent can pattern-match against.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .interfaces import PINNProblem
from .losses import grad


def poisson_1d() -> PINNProblem:
    """u_xx = -pi^2 sin(pi x) on [0, 1], u(0)=u(1)=0. Solution: sin(pi x)."""

    pi = math.pi

    def residual_fn(model: nn.Module, coords: Tensor) -> Tensor:
        u = model(coords)
        u_x = grad(u, coords)
        u_xx = grad(u_x, coords)
        forcing = -(pi**2) * torch.sin(pi * coords)
        return u_xx - forcing

    def boundary_fn(model: nn.Module) -> Tensor:
        edges = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        return model(edges)  # target is 0 at both ends

    def analytical(coords: Tensor) -> Tensor:
        return torch.sin(pi * coords)

    return PINNProblem(
        name="poisson_1d",
        input_dim=1,
        output_dim=1,
        domain=[(0.0, 1.0)],
        residual_fn=residual_fn,
        boundary_fn=boundary_fn,
        analytical=analytical,
    )


def damped_oscillator(omega: float = 2.0, zeta: float = 0.1, t_max: float = 5.0) -> PINNProblem:
    """u'' + 2*zeta*omega*u' + omega^2 u = 0, u(0)=1, u'(0)=0 (underdamped)."""

    def residual_fn(model: nn.Module, coords: Tensor) -> Tensor:
        u = model(coords)
        u_t = grad(u, coords)
        u_tt = grad(u_t, coords)
        return u_tt + 2.0 * zeta * omega * u_t + omega**2 * u

    def boundary_fn(model: nn.Module) -> Tensor:
        t0 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)
        u0 = model(t0)
        u0_t = grad(u0, t0)
        # stack the two initial-condition residuals: u(0)-1 and u'(0)-0
        return torch.cat([u0 - 1.0, u0_t], dim=0)

    def analytical(coords: Tensor) -> Tensor:
        wd = omega * math.sqrt(1.0 - zeta**2)
        decay = torch.exp(-zeta * omega * coords)
        phase = torch.cos(wd * coords) + (zeta * omega / wd) * torch.sin(wd * coords)
        return decay * phase

    return PINNProblem(
        name="damped_oscillator",
        input_dim=1,
        output_dim=1,
        domain=[(0.0, t_max)],
        residual_fn=residual_fn,
        boundary_fn=boundary_fn,
        analytical=analytical,
    )


REFERENCE_PROBLEMS = {
    "poisson_1d": poisson_1d,
    "damped_oscillator": damped_oscillator,
}
