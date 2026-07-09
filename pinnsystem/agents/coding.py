"""Coding agent: ResearchReport -> generated modules -> a clean run.

The LLM emits a set of modules and an entrypoint; this node writes them into the run
workspace, executes the entrypoint in the isolated interpreter, and on failure feeds
the traceback back for a targeted regeneration — the paper's self-debug loop — up to a
budget. On success it collects the run's ``metrics.json`` into `CodeArtifacts`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..execution import RunWorkspace, VenvRunner
from ..state import CodeArtifacts, PINNState
from .base import SupportsStructured, contract_header, invoke_structured, load_prompt

_CONTRACT = contract_header(
    "3 (Coding)",
    prev="a complete ResearchReport (architecture, losses, sampling, hyperparams, data plan)",
    nxt="CodeArtifacts: module->path map, dataset/model/metrics paths, stdout, last_run_error",
)


class GeneratedCode(BaseModel):
    """The LLM's code payload for one attempt."""

    modules: dict[str, str] = Field(
        default_factory=dict, description="filename (e.g. 'model.py') -> full source."
    )
    entrypoint: str = Field("main.py", description="Which module `main` lives in.")
    notes: str = ""


def _write_modules(code: GeneratedCode, workspace: RunWorkspace) -> dict[str, str]:
    written: dict[str, str] = {}
    for filename, source in code.modules.items():
        # LLM-supplied names: strip directory components so a module can't be written
        # outside the run's scripts dir (absolute/`..` paths escape a bare join).
        safe_name = Path(filename).name
        if not safe_name:
            continue
        path = workspace.scripts / safe_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        written[safe_name] = str(path)
    return written


def _collect_metrics(workspace: RunWorkspace) -> tuple[str, dict]:
    """Find metrics.json anywhere the entrypoint might have written it."""

    for candidate in (workspace.root / "metrics.json", workspace.metrics / "metrics.json"):
        if candidate.exists():
            try:
                return str(candidate), json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return str(candidate), {}
    return "", {}


def coding_node(
    state: PINNState,
    llm: SupportsStructured,
    *,
    workspace: RunWorkspace,
    runner: VenvRunner | None = None,
    code_debug_budget: int = 5,
    run_timeout: float = 600.0,
) -> dict:
    """Generate, run, and self-debug PINN modules; return `CodeArtifacts`."""

    runner = runner or VenvRunner()
    research = state["research"]
    directive = ""
    if state.get("feedback"):
        directive = state["feedback"].directive  # targeted regeneration on revise

    system = f"{load_prompt('coding')}\n\n{_CONTRACT}"
    base_human = (
        f"ResearchReport:\n{research.model_dump_json(indent=2)}\n\n"
        f"Write modules under the scripts dir; entrypoint must write metrics.json to the "
        f"current working directory. Reuse `pinnsystem.pinn` helpers where possible.\n"
        + (f"\nFeedback directive to address: {directive}\n" if directive else "")
    )

    last: CodeArtifacts | None = None
    error_context = ""

    for _ in range(max(1, code_debug_budget)):
        human = base_human + error_context
        code: GeneratedCode = invoke_structured(llm, GeneratedCode, system, human)
        modules = _write_modules(code, workspace)

        entry_name = Path(code.entrypoint).name or "main.py"
        entry_path = Path(modules.get(entry_name, workspace.scripts / entry_name))
        outcome = runner.run_script(entry_path, cwd=workspace.root, timeout=run_timeout)

        metrics_path, _metrics = _collect_metrics(workspace)
        last = CodeArtifacts(
            modules=modules,
            dataset_path=str(workspace.data),
            model_path=str(workspace.models),
            metrics_path=metrics_path,
            last_run_stdout=outcome.stdout,
            last_run_error=outcome.error,
        )

        if outcome.ok and metrics_path:
            return {"code": last}

        # Feed the failure back for a targeted fix on the next attempt.
        error_context = (
            f"\n\nPREVIOUS ATTEMPT FAILED. Fix the responsible module only.\n"
            f"stderr:\n{outcome.error}\n"
            f"stdout tail:\n{outcome.stdout[-1000:]}\n"
        )

    return {"code": last}
