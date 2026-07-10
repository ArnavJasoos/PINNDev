"""Shared state schema — the single source of truth passed between all agents.

Pydantic models carry the structured payloads; ``PINNState`` is the TypedDict the
LangGraph ``StateGraph`` threads through every node. Keeping this module dependency
free (only pydantic + stdlib) means every other layer can import it without pulling
in torch, langgraph, or the MCP stack.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Problem specification (Parser output)
# --------------------------------------------------------------------------- #


class DomainSpec(BaseModel):
    """Spatial/temporal domain the PDE lives on."""

    dims: int = Field(..., ge=1, description="Number of independent variables.")
    variables: list[str] = Field(
        default_factory=list, description="Independent variable names, e.g. ['t'] or ['x', 't']."
    )
    bounds: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="Per-variable (low, high) bounds."
    )
    geometry_complexity: float = Field(
        0.0, ge=0.0, le=1.0, description="0 = rectangular/interval, 1 = irregular geometry."
    )


class SymbolicPDE(BaseModel):
    """Canonical PDE/ODE representation with a feature vector for arch matching."""

    latex: str = ""
    sympy_src: str = Field(
        "", description="Reconstructable sympy expression string for the residual == 0."
    )
    operators: list[str] = Field(
        default_factory=list, description="e.g. ['d/dt', 'd2/dx2', 'nonlinear_prod']."
    )
    boundary_conditions: list[str] = Field(default_factory=list)
    initial_conditions: list[str] = Field(default_factory=list)
    feature_vector: dict[str, float] = Field(
        default_factory=dict,
        description="phi(E): periodicity, geometry_complexity, multiscale in [0,1].",
    )


class ProblemSpec(BaseModel):
    """Normalized, user-approved problem statement (Parser -> Research contract)."""

    raw_query: str
    normalized_statement: str = ""
    pde: Optional[SymbolicPDE] = None
    domain: Optional[DomainSpec] = None
    quantities: list[str] = Field(default_factory=list)
    user_provided_formulas: bool = False
    user_provided_dataset: bool = False
    dataset_path: Optional[str] = None
    approved_by_user: bool = False


# --------------------------------------------------------------------------- #
# Research report (Research output)
# --------------------------------------------------------------------------- #


class LossTerm(BaseModel):
    name: Literal["residual", "bc", "ic", "data"]
    weight: float = 1.0
    description: str = ""


class SamplingPlan(BaseModel):
    collocation_points: int = 2000
    boundary_points: int = 200
    initial_points: int = 200
    adaptive: bool = False


class HyperParams(BaseModel):
    width: int = 64
    depth: int = 4
    lr: float = 1e-3
    epochs: int = 5000
    optimizer: Literal["adam", "lbfgs"] = "adam"
    activation: Literal["tanh", "sin", "relu", "gelu"] = "tanh"


class DataGenPlan(BaseModel):
    method: Literal["solve_ivp", "finite_difference", "py_pde", "analytical", "user_dataset"]
    params: dict[str, Any] = Field(default_factory=dict)
    n_points: int = 200


class ResearchReport(BaseModel):
    """Self-contained plan a coder implements with zero extra research."""

    architecture: Literal[
        "MLP", "Fourier-MLP", "SIREN", "CNN", "GNN", "Transformer"
    ] = "MLP"
    arch_rationale: str = ""
    loss_terms: list[LossTerm] = Field(default_factory=list)
    sampling: SamplingPlan = Field(default_factory=SamplingPlan)
    hyperparams: HyperParams = Field(default_factory=HyperParams)
    data_generation: Optional[DataGenPlan] = None
    citations: list[str] = Field(default_factory=list)
    forbidden_approaches: list[str] = Field(
        default_factory=list, description="Failure memory — approaches Research must not reuse."
    )


# --------------------------------------------------------------------------- #
# Code artifacts (Coding output)
# --------------------------------------------------------------------------- #


class CodeArtifacts(BaseModel):
    modules: dict[str, str] = Field(default_factory=dict, description="module_name -> filepath.")
    dataset_path: str = ""
    model_path: str = ""
    metrics_path: str = ""
    last_run_stdout: str = ""
    last_run_error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Feedback verdict (Feedback output)
# --------------------------------------------------------------------------- #


class QualityMetrics(BaseModel):
    mse: float = float("inf")
    rel_l2: float = float("inf")
    convergence_iters: int = 0
    loss_smoothness: float = 0.0


class FeedbackVerdict(BaseModel):
    quality_score: float = 0.0
    metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    passed_threshold: bool = False
    decision: Literal["accept", "revise_code", "revert_research", "await_user"] = "revise_code"
    faulty_module: Optional[str] = None
    directive: str = ""
    plots: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #


class PINNState(TypedDict, total=False):
    """The object threaded through every LangGraph node.

    ``total=False`` so partial updates from a node merge cleanly; nodes return
    only the keys they change.
    """

    spec: ProblemSpec
    research: Optional[ResearchReport]
    code: Optional[CodeArtifacts]
    feedback: Optional[FeedbackVerdict]
    history: list[dict]
    iteration: int
    max_iterations: int
    accuracy_threshold: float
    messages: list
    pending_user_action: Optional[str]
    # Mid-session iteration (smart re-route): ``followup`` is the raw user message that
    # re-enters a completed run; the intent router classifies it into a stage and passes
    # the instruction on as ``revision_note`` (consumed by research/coding).
    followup: Optional[str]
    followup_target: Optional[str]
    revision_note: Optional[str]


def new_state(
    raw_query: str,
    *,
    accuracy_threshold: float = 1e-3,
    max_iterations: int = 3,
) -> PINNState:
    """Construct a fresh state for a run."""

    return PINNState(
        spec=ProblemSpec(raw_query=raw_query),
        research=None,
        code=None,
        feedback=None,
        history=[],
        iteration=0,
        max_iterations=max_iterations,
        accuracy_threshold=accuracy_threshold,
        messages=[],
        pending_user_action=None,
        followup=None,
        followup_target=None,
        revision_note=None,
    )
