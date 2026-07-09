"""Module I/O contracts for the PINN core.

These dataclasses are the interfaces the Coding agent's generated modules must honor
(model / pde_loss / data / train_loop / validation). Keeping them here — rather than
inlined in each module — lets the symbolic checker and the agents reason about the
boundaries between modules explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor, nn

# A residual function maps (model, interior_coords) -> pointwise PDE residual (N, out_dim).
ResidualFn = Callable[[nn.Module, Tensor], Tensor]

# A boundary function maps model -> scalar-reducible boundary/initial residual tensor.
BoundaryFn = Callable[[nn.Module], Tensor]

# An analytical solution maps coords (N, in_dim) -> reference solution (N, out_dim).
AnalyticalFn = Callable[[Tensor], Tensor]


@dataclass
class PINNProblem:
    """A fully-specified PINN problem the training loop can consume directly.

    ``residual_fn`` encodes the PDE/ODE (residual == 0). ``boundary_fn`` returns a
    residual tensor for the BC/IC constraints. ``analytical`` (when known) provides
    ground truth for evaluation and for generating hard-computed test data.
    """

    name: str
    input_dim: int
    output_dim: int
    domain: list[tuple[float, float]]
    residual_fn: ResidualFn
    boundary_fn: BoundaryFn
    analytical: Optional[AnalyticalFn] = None

    def sample_interior(self, n: int, *, generator: Optional[torch.Generator] = None) -> Tensor:
        """Uniformly sample ``n`` collocation points inside the domain."""

        lows = torch.tensor([b[0] for b in self.domain], dtype=torch.float32)
        highs = torch.tensor([b[1] for b in self.domain], dtype=torch.float32)
        unit = torch.rand(n, self.input_dim, generator=generator)
        return lows + unit * (highs - lows)

    def test_grid(self, n: int) -> Tensor:
        """Deterministic evaluation grid.

        1-D: a linspace over the single axis. Higher dims: a low-discrepancy-ish
        uniform grid (kept simple — the smoke targets are 1-D).
        """

        if self.input_dim == 1:
            lo, hi = self.domain[0]
            return torch.linspace(lo, hi, n, dtype=torch.float32).unsqueeze(1)

        per_axis = max(2, int(round(n ** (1.0 / self.input_dim))))
        axes = [torch.linspace(lo, hi, per_axis, dtype=torch.float32) for lo, hi in self.domain]
        mesh = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([m.reshape(-1) for m in mesh], dim=1)


@dataclass
class TrainResult:
    """What the training loop reports back (train_loop -> validation contract)."""

    model: nn.Module
    loss_history: list[float] = field(default_factory=list)
    residual_history: list[float] = field(default_factory=list)
    boundary_history: list[float] = field(default_factory=list)
    converged_iters: int = 0
    final_loss: float = float("inf")
