"""Knowledge-guided architecture matching: phi(E) . psi(A).

Each architecture carries a capability vector psi(A) over the same axes as a PDE's
feature vector phi(E) — ``periodicity``, ``geometry_complexity``, ``multiscale``. The
Research agent scores architectures by weighted cosine similarity and picks the
argmax, prioritizing multi-scale > geometry > periodicity (the paper's ordering).
"""

from __future__ import annotations

import math
from typing import Any

_AXES = ("periodicity", "geometry_complexity", "multiscale")

# Priority weights on the matching axes (multiscale dominates).
_AXIS_WEIGHTS = {"periodicity": 1.0, "geometry_complexity": 1.5, "multiscale": 2.0}

# psi(A): each architecture's relative strength on each axis, in [0, 1].
ARCH_CAPABILITIES: dict[str, dict[str, float]] = {
    "MLP": {"periodicity": 0.3, "geometry_complexity": 0.3, "multiscale": 0.2},
    "Fourier-MLP": {"periodicity": 0.9, "geometry_complexity": 0.4, "multiscale": 0.9},
    "SIREN": {"periodicity": 1.0, "geometry_complexity": 0.4, "multiscale": 0.8},
    "CNN": {"periodicity": 0.4, "geometry_complexity": 0.7, "multiscale": 0.5},
    "GNN": {"periodicity": 0.3, "geometry_complexity": 1.0, "multiscale": 0.5},
    "Transformer": {"periodicity": 0.5, "geometry_complexity": 0.7, "multiscale": 0.7},
}


def _weighted_cosine(phi: dict[str, float], psi: dict[str, float]) -> float:
    num = 0.0
    phi_norm = 0.0
    psi_norm = 0.0
    for axis in _AXES:
        w = _AXIS_WEIGHTS[axis]
        a = w * float(phi.get(axis, 0.0))
        b = w * float(psi.get(axis, 0.0))
        num += a * b
        phi_norm += a * a
        psi_norm += b * b
    denom = math.sqrt(phi_norm) * math.sqrt(psi_norm)
    return num / denom if denom > 0 else 0.0


def match_architecture(
    feature_vector: dict[str, float],
    *,
    forbidden: list[str] | None = None,
) -> dict[str, Any]:
    """Return the best architecture for a PDE feature vector.

    ``forbidden`` architectures (failure memory) are excluded from the argmax. If the
    feature vector is empty/zero, MLP is the safe default. Returns the choice, a
    rationale string, and the full ranked score table.
    """

    forbidden_set = {f.strip() for f in (forbidden or [])}
    scores = {
        arch: _weighted_cosine(feature_vector, psi)
        for arch, psi in ARCH_CAPABILITIES.items()
        if arch not in forbidden_set
    }
    if not scores or all(v == 0.0 for v in scores.values()):
        default = "MLP" if "MLP" not in forbidden_set else next(iter(scores), "MLP")
        return {
            "architecture": default,
            "rationale": "No discriminating PDE features; defaulting to a plain MLP backbone.",
            "scores": scores,
        }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    rationale = (
        f"phi(E).psi(A) weighted-cosine argmax selects {best} (score={best_score:.3f}); "
        f"runner-up {ranked[1][0]} ({ranked[1][1]:.3f})."
        if len(ranked) > 1
        else f"{best} selected (score={best_score:.3f})."
    )
    return {"architecture": best, "rationale": rationale, "scores": dict(ranked)}
