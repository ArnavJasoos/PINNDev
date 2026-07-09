"""Network architectures for PINNs.

The Research agent selects one of these by name (``ResearchReport.architecture``);
:func:`build_network` is the single construction entry point so the Coding agent's
generated ``model`` module can stay a thin call into here.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

Activation = Literal["tanh", "sin", "relu", "gelu"]


class Sine(nn.Module):
    """Sine activation with a frequency scale (SIREN)."""

    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * x)


def _activation(name: Activation) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return Sine()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name!r}")


class MLP(nn.Module):
    """Plain fully-connected network — the default PINN backbone."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 4,
        activation: Activation = "tanh",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), _activation(activation)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), _activation(activation)]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FourierMLP(nn.Module):
    """MLP with a fixed random Fourier feature input mapping.

    Mitigates spectral bias on multi-scale / oscillatory targets by lifting inputs
    into ``[sin, cos]`` of random projections before the dense stack.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 4,
        activation: Activation = "tanh",
        n_features: int = 64,
        sigma: float = 2.0,
    ) -> None:
        super().__init__()
        b = torch.randn(in_dim, n_features) * sigma
        self.register_buffer("B", b)
        mapped_dim = 2 * n_features
        self.mlp = MLP(mapped_dim, out_dim, width=width, depth=depth, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        proj = 2.0 * math.pi * x @ self.B
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.mlp(feats)


class SIREN(nn.Module):
    """Sinusoidal-representation network with the principled SIREN init."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 4,
        w0: float = 30.0,
        w0_hidden: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()

        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self._siren_init(w0, w0_hidden)

        # sine on every layer except the output
        for i in range(len(self.layers) - 1):
            self.acts.append(Sine(w0 if i == 0 else w0_hidden))

    def _siren_init(self, w0: float, w0_hidden: float) -> None:
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                fan_in = layer.weight.shape[1]
                if i == 0:
                    bound = 1.0 / fan_in
                else:
                    bound = math.sqrt(6.0 / fan_in) / w0_hidden
                layer.weight.uniform_(-bound, bound)
                if layer.bias is not None:
                    layer.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = self.acts[i](layer(x))
        return self.layers[-1](x)


def build_network(
    architecture: str,
    in_dim: int,
    out_dim: int,
    *,
    width: int = 64,
    depth: int = 4,
    activation: Activation = "tanh",
    seed: int | None = None,
) -> nn.Module:
    """Construct a network by name (matches ``ResearchReport.architecture``)."""

    if seed is not None:
        torch.manual_seed(seed)

    arch = architecture.lower()
    if arch in {"mlp"}:
        return MLP(in_dim, out_dim, width=width, depth=depth, activation=activation)
    if arch in {"fourier-mlp", "fourier_mlp", "fouriermlp"}:
        return FourierMLP(in_dim, out_dim, width=width, depth=depth, activation=activation)
    if arch in {"siren"}:
        return SIREN(in_dim, out_dim, width=width, depth=depth)

    # CNN/GNN/Transformer are declared in the schema for future problem classes; for
    # the point-wise collocation problems here they degrade to the MLP backbone.
    if arch in {"cnn", "gnn", "transformer"}:
        return MLP(in_dim, out_dim, width=width, depth=depth, activation=activation)

    raise ValueError(f"Unknown architecture: {architecture!r}")
