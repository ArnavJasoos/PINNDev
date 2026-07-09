"""Entry point: ``python -m pinnsystem`` launches the NiceGUI app.

The GUI (Build Order phase 6) is not implemented yet; until then this reports what
is available so the entry point never dangles.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from .gui.app import run  # noqa: F401  (built in phase 6)
    except ModuleNotFoundError:
        sys.stderr.write(
            "The NiceGUI front-end is not built yet (Build Order phase 6).\n"
            "The PINN core and state layer are ready - see tests/ for usage:\n"
            "    python -m pytest\n"
        )
        return 1

    return run()  # type: ignore[no-any-return]


if __name__ == "__main__":
    raise SystemExit(main())
