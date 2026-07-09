"""Sympy-backed parsing and PDE-loss equivalence checking.

The Parser uses :func:`sympy_parse` to canonicalize a PDE; the Coding agent's
symbolic check uses :func:`symbolic_equivalence` to verify a generated ``pde_loss``
really encodes the Research agent's target PDE. Mirroring the paper, equivalence is a
*composite* signal — exact symbolic simplification first, then a numeric-sampling
score — so strict symbolic checks don't over-reject algebraically-equal forms.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import sympy as sp

# sympify() calls eval(); an attacker-influenced string could reach arbitrary objects
# via dunder attribute chains (e.g. ().__class__...). Legitimate PDE/ODE expressions
# never contain these tokens, so we reject them before sympify as defense-in-depth.
_FORBIDDEN_TOKENS = ("__", "import", "lambda", "exec", "eval", "os.", "sys.", "`")


def _reject_unsafe(expr_src: str) -> Optional[str]:
    lowered = expr_src.lower()
    for token in _FORBIDDEN_TOKENS:
        if token in lowered:
            return f"rejected unsafe token {token!r} in expression"
    return None


def sympy_parse(expr_src: str) -> dict[str, Any]:
    """Parse a sympy expression string into a canonical form + metadata.

    Returns ``{ok, latex, symbols, canonical, error}``. Never raises — a parse
    failure (or a rejected unsafe token) comes back as ``ok=False`` with the reason.
    """

    unsafe = _reject_unsafe(expr_src)
    if unsafe:
        return {"ok": False, "latex": "", "symbols": [], "canonical": "", "error": unsafe}

    try:
        expr = sp.sympify(expr_src)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        return {"ok": False, "latex": "", "symbols": [], "canonical": "", "error": str(exc)}

    return {
        "ok": True,
        "latex": sp.latex(expr),
        "symbols": sorted(str(s) for s in expr.free_symbols),
        "canonical": str(sp.simplify(expr)),
        "error": None,
    }


def symbolic_equivalence(
    lhs: str,
    rhs: str,
    *,
    n_samples: int = 12,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Composite equivalence score in [0, 1] between two residual expressions.

    ``equivalent`` is True when either sympy proves ``lhs - rhs == 0`` or the numeric
    agreement fraction over random sample points meets a high bar. ``score`` is that
    agreement fraction (1.0 when symbolically proven).
    """

    unsafe = _reject_unsafe(lhs) or _reject_unsafe(rhs)
    if unsafe:
        return {"equivalent": False, "score": 0.0, "method": "rejected", "error": unsafe}

    try:
        a = sp.sympify(lhs)
        b = sp.sympify(rhs)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        return {"equivalent": False, "score": 0.0, "method": "parse_error", "error": str(exc)}

    diff = sp.simplify(a - b)
    if diff == 0:
        return {"equivalent": True, "score": 1.0, "method": "symbolic", "error": None}

    symbols = sorted(a.free_symbols | b.free_symbols, key=str)
    if not symbols:
        try:
            equal = abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            equal = False
        return {
            "equivalent": equal,
            "score": 1.0 if equal else 0.0,
            "method": "constant",
            "error": None,
        }

    score = _numeric_agreement(diff, symbols, n_samples=n_samples, tol=tol)
    return {
        "equivalent": score >= 0.99,
        "score": score,
        "method": "numeric",
        "error": None,
    }


def _numeric_agreement(diff: sp.Expr, symbols: list[sp.Symbol], *, n_samples: int, tol: float) -> float:
    """Fraction of random points where ``diff`` evaluates to ~0."""

    rng = random.Random(0)
    f = sp.lambdify(symbols, diff, "math")
    hits = 0
    evaluated = 0
    for _ in range(n_samples):
        point = [rng.uniform(-2.0, 2.0) for _ in symbols]
        try:
            val = f(*point)
        except (ValueError, ZeroDivisionError, OverflowError, ArithmeticError):
            continue
        evaluated += 1
        if abs(complex(val)) <= tol:
            hits += 1
    if evaluated == 0:
        return 0.0
    return hits / evaluated
