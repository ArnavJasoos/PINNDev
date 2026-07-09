"""Deterministic PINN operations exposed as tools.

Thin, side-effect-explicit wrappers over the Phase-2 PINN core. Agents mostly drive
training through model-written scripts (via the venv runner), but these give the
Feedback agent its plots and provide a dependable path for the skip-branches
(dataset/formulas provided) where no code generation is needed.

All functions take/return JSON-friendly primitives and file paths so they map cleanly
onto MCP tool signatures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from ..pinn.architectures import build_network
from ..pinn.evaluate import evaluate, quality_score
from ..pinn.problems import REFERENCE_PROBLEMS
from ..pinn.solvers import (
    finite_difference_poisson_1d,
    harmonic_oscillator_reference,
    poisson_1d_manufactured,
)
from ..pinn.train import train_pinn
from ..state import HyperParams, SamplingPlan

# Named hard-computed data generators (DataGenPlan -> concrete numpy solve).
_GENERATORS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "harmonic_oscillator": harmonic_oscillator_reference,
    "poisson_1d": poisson_1d_manufactured,
    "finite_difference_poisson": finite_difference_poisson_1d,
}


def build_dataset(
    generator: str,
    out_path: str | Path,
    *,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Numerically solve a reference ODE/PDE and save ``(inputs, targets)`` as ``.npz``.

    ``generator`` selects a solver in :data:`_GENERATORS`; ``params`` are forwarded to
    it. Returns the saved path, point count, and array shapes.
    """

    if generator not in _GENERATORS:
        raise ValueError(
            f"Unknown generator {generator!r}; choose from {sorted(_GENERATORS)}."
        )

    fn = _GENERATORS[generator]
    inputs, targets = fn(**(params or {}))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, inputs=inputs, targets=targets)
    return {
        "path": str(out),
        "n_points": int(inputs.shape[0]),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
    }


def train_run(
    problem_name: str,
    out_dir: str | Path,
    *,
    hyperparams: Optional[dict[str, Any]] = None,
    sampling: Optional[dict[str, Any]] = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a PINN on a reference problem and persist the model + its arch config.

    Returns paths and convergence stats. The arch sidecar lets :func:`evaluate_run`
    rebuild the network without re-supplying its shape.
    """

    if problem_name not in REFERENCE_PROBLEMS:
        raise ValueError(
            f"Unknown problem {problem_name!r}; choose from {sorted(REFERENCE_PROBLEMS)}."
        )

    problem = REFERENCE_PROBLEMS[problem_name]()
    hp = HyperParams(**(hyperparams or {}))
    sp = SamplingPlan(**(sampling or {}))
    architecture = (hyperparams or {}).get("architecture", "MLP")

    model = build_network(
        architecture,
        problem.input_dim,
        problem.output_dim,
        width=hp.width,
        depth=hp.depth,
        activation=hp.activation,
        seed=seed,
    )
    result = train_pinn(problem, hp, sp, model=model, seed=seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"{problem_name}_model.pt"
    torch.save(result.model.state_dict(), model_path)

    config = {
        "problem": problem_name,
        "architecture": architecture,
        "input_dim": problem.input_dim,
        "output_dim": problem.output_dim,
        "width": hp.width,
        "depth": hp.depth,
        "activation": hp.activation,
        "seed": seed,
    }
    config_path = out / f"{problem_name}_arch.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    loss_path = out / f"{problem_name}_loss.json"
    loss_path.write_text(json.dumps(result.loss_history), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "loss_history_path": str(loss_path),
        "final_loss": result.final_loss,
        "converged_iters": result.converged_iters,
    }


def _rebuild_model(config: dict[str, Any], model_path: str | Path) -> torch.nn.Module:
    model = build_network(
        config["architecture"],
        config["input_dim"],
        config["output_dim"],
        width=config["width"],
        depth=config["depth"],
        activation=config["activation"],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def evaluate_run(
    problem_name: str,
    model_path: str | Path,
    config_path: str | Path,
    *,
    loss_history_path: str | Path | None = None,
    n_test: int = 400,
) -> dict[str, Any]:
    """Evaluate a saved model against ground truth and compute the quality score."""

    from ..pinn.interfaces import TrainResult  # local import: torch types only

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = _rebuild_model(config, model_path)

    loss_history: list[float] = []
    if loss_history_path and Path(loss_history_path).exists():
        loss_history = json.loads(Path(loss_history_path).read_text(encoding="utf-8"))

    problem = REFERENCE_PROBLEMS[problem_name]()
    result = TrainResult(model=model, loss_history=loss_history)
    report = evaluate(problem, result, n_test=n_test)
    score = quality_score(report)

    return {
        "mse": report.mse,
        "rel_l2": report.rel_l2,
        "loss_smoothness": report.loss_smoothness,
        "convergence_iters": report.convergence_iters,
        "quality_score": score,
        "test_inputs": report.test_inputs.tolist(),
        "prediction": report.prediction.tolist(),
        "reference": report.reference.tolist() if report.reference is not None else None,
    }


def plot_results(
    test_inputs: list | np.ndarray,
    prediction: list | np.ndarray,
    out_path: str | Path,
    *,
    reference: list | np.ndarray | None = None,
    title: str = "PINN prediction vs ground truth",
) -> dict[str, Any]:
    """Save a prediction-vs-truth plot.

    Uses matplotlib when available; otherwise falls back to an ``.npz`` dump of the
    arrays so the pipeline never hard-fails on a missing plotting backend.
    """

    x = np.asarray(test_inputs, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(reference, dtype=float) if reference is not None else None
    x_axis = x[:, 0] if x.ndim > 1 else x

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_axis, pred[:, 0] if pred.ndim > 1 else pred, label="PINN", lw=2)
        if ref is not None:
            ax.plot(
                x_axis,
                ref[:, 0] if ref.ndim > 1 else ref,
                label="ground truth",
                ls="--",
                lw=2,
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return {"path": str(out), "backend": "matplotlib"}
    except ImportError:
        fallback = out.with_suffix(".npz")
        np.savez(fallback, test_inputs=x, prediction=pred, reference=ref if ref is not None else [])
        return {"path": str(fallback), "backend": "npz-fallback"}
