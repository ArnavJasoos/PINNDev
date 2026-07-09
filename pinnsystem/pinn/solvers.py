"""Hard-computed data generation.

Numerically (or analytically) solve the ODE/PDE to produce train/test ground truth,
rather than reusing a benchmark dataset. The Research agent's ``DataGenPlan`` selects
a method here; the functions return plain numpy arrays so they are trivially
serializable into a run's dataset artifact.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


def solve_ode_ivp(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    n_points: int = 200,
    *,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a first-order ODE system and sample it on a uniform grid.

    Returns ``(t, y)`` where ``t`` is ``(n_points,)`` and ``y`` is ``(n_points, dim)``.
    """

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y.T


def harmonic_oscillator_reference(
    omega: float = 2.0,
    zeta: float = 0.1,
    t_max: float = 5.0,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Damped harmonic oscillator ``u'' + 2 zeta omega u' + omega^2 u = 0``.

    Initial conditions ``u(0) = 1``, ``u'(0) = 0``. Returns ``(t, u)``.
    """

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        u, v = y
        return np.array([v, -2.0 * zeta * omega * v - omega**2 * u])

    t, y = solve_ode_ivp(rhs, (0.0, t_max), np.array([1.0, 0.0]), n_points=n_points)
    return t, y[:, 0:1]


def poisson_1d_manufactured(
    n_points: int = 200,
    domain: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """1-D Poisson ``u_xx = -pi^2 sin(pi x)`` with ``u(0)=u(1)=0``.

    Manufactured analytical solution ``u(x) = sin(pi x)``. Returns ``(x, u)``.
    """

    x = np.linspace(domain[0], domain[1], n_points)
    u = np.sin(np.pi * x)
    return x.reshape(-1, 1), u.reshape(-1, 1)


def finite_difference_poisson_1d(
    forcing: Callable[[np.ndarray], np.ndarray],
    n_points: int = 200,
    domain: tuple[float, float] = (0.0, 1.0),
    bc: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``u_xx = f`` on an interval with Dirichlet BCs via a tridiagonal solve.

    A concrete ``finite_difference`` data-gen backend for problems without a closed
    form. Returns ``(x, u)``.
    """

    x = np.linspace(domain[0], domain[1], n_points)
    h = x[1] - x[0]
    n_inner = n_points - 2

    a = np.zeros((n_inner, n_inner))
    np.fill_diagonal(a, -2.0)
    idx = np.arange(n_inner - 1)
    a[idx, idx + 1] = 1.0
    a[idx + 1, idx] = 1.0

    rhs = forcing(x[1:-1]) * h**2
    rhs[0] -= bc[0]
    rhs[-1] -= bc[1]

    u_inner = np.linalg.solve(a, rhs)
    u = np.empty(n_points)
    u[0], u[-1] = bc
    u[1:-1] = u_inner
    return x.reshape(-1, 1), u.reshape(-1, 1)
