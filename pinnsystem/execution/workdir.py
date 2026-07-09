"""Per-run scratch directories and artifact paths.

Every run gets one :class:`RunWorkspace` rooted at ``<runs_dir>/<run_id>/`` with a
fixed subdirectory layout. Agents write generated scripts, datasets, models, plots,
and metrics into the corresponding subdir so nothing leaks between runs and the GUI
can locate artifacts deterministically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

_SUBDIRS = ("scripts", "data", "models", "plots", "metrics")


@dataclass(frozen=True)
class RunWorkspace:
    """Filesystem layout for a single run."""

    run_id: str
    root: Path

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    def path(self, *parts: str) -> Path:
        """Resolve a path inside this workspace, creating parent dirs as needed."""

        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def new_workspace(runs_dir: str | Path = "runs", run_id: str | None = None) -> RunWorkspace:
    """Create (or reuse) a run workspace with all standard subdirectories present."""

    rid = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    root = Path(runs_dir) / rid
    for sub in _SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return RunWorkspace(run_id=rid, root=root)
