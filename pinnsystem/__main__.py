"""Entry point: ``python -m pinnsystem`` launches the NiceGUI app.

The GUI needs the optional GUI + agent extras. When they are missing we report what to
install rather than dumping a traceback, so the entry point never dangles.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from pinnsystem.gui.app import run
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
        _report_missing(exc)
        return 1

    try:
        return run()
    except ModuleNotFoundError as exc:
        _report_missing(exc)
        return 1


def _report_missing(exc: ModuleNotFoundError) -> None:
    sys.stderr.write(
        f"The GUI needs an optional dependency that is not installed ({exc.name}).\n"
        "Install the front-end + agent stack:\n"
        "    pip install -e \".[agents,gui]\"\n\n"
        "The PINN core, tools, and agents are usable without the GUI — see tests/:\n"
        "    python -m pytest\n"
    )


if __name__ in {"__main__", "__mp_main__"}:
    # NiceGUI re-runs this file (by path) per page request to render the auto-index.
    # On that re-run ui.run() is a no-op, so main() returns; raising SystemExit here
    # would propagate into NiceGUI's page renderer. Only exit on a real dependency
    # failure (main() returns non-zero before the server ever starts).
    _rc = main()
    if _rc:
        raise SystemExit(_rc)
