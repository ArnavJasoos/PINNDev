"""Isolated subprocess execution of model-written Python.

The Coding agent generates scripts that must run somewhere safe. ``VenvRunner`` runs
them through a dedicated interpreter as a plain ``subprocess`` — no shell string
interpolation — with a hard timeout and captured stdout/stderr.

By default the runner reuses the current interpreter (fast, and already has torch/
numpy). Call :meth:`ensure_venv` to build a dedicated venv; ``--system-site-packages``
keeps heavy scientific deps visible without a multi-minute reinstall while still
giving a separate interpreter and site directory.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunOutcome:
    """Result of running a script/snippet."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def error(self) -> Optional[str]:
        """A single error string for the Feedback agent, or ``None`` on success."""

        if self.timed_out:
            return "TimeoutExpired: script exceeded the time budget."
        if self.returncode != 0:
            return self.stderr.strip() or f"Non-zero exit: {self.returncode}"
        return None


class VenvRunner:
    """Runs Python scripts in an isolated interpreter with a timeout."""

    def __init__(self, python_exe: str | Path | None = None) -> None:
        self.python_exe = str(python_exe) if python_exe else sys.executable

    @classmethod
    def ensure_venv(
        cls,
        venv_dir: str | Path,
        *,
        system_site_packages: bool = True,
    ) -> "VenvRunner":
        """Create a venv at ``venv_dir`` if absent and return a runner bound to it."""

        vdir = Path(venv_dir)
        python_exe = _venv_python(vdir)
        if not python_exe.exists():
            builder = venv.EnvBuilder(
                system_site_packages=system_site_packages,
                with_pip=True,
                clear=False,
            )
            builder.create(str(vdir))
        return cls(python_exe)

    def run_script(
        self,
        script_path: str | Path,
        *,
        cwd: str | Path | None = None,
        timeout: float = 300.0,
        args: Optional[list[str]] = None,
    ) -> RunOutcome:
        """Run an existing script file and capture its output."""

        cmd = [self.python_exe, str(script_path), *(args or [])]
        return self._run(cmd, cwd=cwd, timeout=timeout)

    def run_code(
        self,
        code: str,
        *,
        workdir: str | Path,
        filename: str = "_snippet.py",
        timeout: float = 300.0,
    ) -> RunOutcome:
        """Write ``code`` into ``workdir`` and run it."""

        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        script = wd / filename
        script.write_text(code, encoding="utf-8")
        return self.run_script(script, cwd=wd, timeout=timeout)

    def _run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path | None,
        timeout: float,
    ) -> RunOutcome:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RunOutcome(
                returncode=-1,
                stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr or "" if isinstance(exc.stderr, str) else "",
                timed_out=True,
            )
        except OSError as exc:
            # Misconfigured interpreter / missing cwd: fail into the RunOutcome path
            # rather than crashing the calling agent node.
            return RunOutcome(returncode=-1, stdout="", stderr=str(exc), timed_out=False)
        return RunOutcome(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
        )


def _venv_python(venv_dir: Path) -> Path:
    """Path to the interpreter inside a venv, cross-platform."""

    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"
