"""MCP server: PINN tools (build_dataset, train, evaluate, plot).

Deterministic wrappers used directly by the skip-branches and by the Feedback agent
for plotting. Heavy work is delegated to :mod:`pinnsystem.tools.pinn_ops`.
"""

from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..tools import build_dataset, evaluate_run, plot_results, train_run

mcp = FastMCP("pinn_tools")


@mcp.tool()
def build_dataset_tool(
    generator: str,
    out_path: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Numerically solve a reference ODE/PDE and save the dataset as ``.npz``."""

    return build_dataset(generator, out_path, params=params)


@mcp.tool()
def train_pinn_tool(
    problem_name: str,
    out_dir: str,
    hyperparams: Optional[dict[str, Any]] = None,
    sampling: Optional[dict[str, Any]] = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a PINN on a reference problem; persist model + arch config."""

    return train_run(problem_name, out_dir, hyperparams=hyperparams, sampling=sampling, seed=seed)


@mcp.tool()
def evaluate_pinn_tool(
    problem_name: str,
    model_path: str,
    config_path: str,
    loss_history_path: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate a saved model and compute the composite quality score."""

    return evaluate_run(
        problem_name, model_path, config_path, loss_history_path=loss_history_path
    )


@mcp.tool()
def plot_results_tool(
    test_inputs: list,
    prediction: list,
    out_path: str,
    reference: Optional[list] = None,
    title: str = "PINN prediction vs ground truth",
) -> dict[str, Any]:
    """Save a prediction-vs-truth plot (matplotlib, npz fallback)."""

    return plot_results(test_inputs, prediction, out_path, reference=reference, title=title)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
