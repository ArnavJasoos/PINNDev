"""Data-generation solver correctness."""

import numpy as np

from pinnsystem.pinn.solvers import (
    finite_difference_poisson_1d,
    harmonic_oscillator_reference,
    poisson_1d_manufactured,
)


def test_poisson_manufactured_solution():
    x, u = poisson_1d_manufactured(n_points=101)
    assert x.shape == (101, 1)
    assert np.allclose(u[0], 0.0, atol=1e-12)
    assert np.allclose(u[-1], 0.0, atol=1e-12)
    # peak of sin(pi x) at x=0.5
    assert np.isclose(u.max(), 1.0, atol=1e-3)


def test_finite_difference_matches_analytical():
    # u_xx = -pi^2 sin(pi x) -> u = sin(pi x)
    forcing = lambda x: -(np.pi**2) * np.sin(np.pi * x)
    x, u = finite_difference_poisson_1d(forcing, n_points=201)
    ref = np.sin(np.pi * x)
    assert np.max(np.abs(u - ref)) < 1e-3


def test_harmonic_oscillator_initial_conditions():
    t, u = harmonic_oscillator_reference(n_points=200)
    assert t.shape == (200,)
    assert np.isclose(u[0, 0], 1.0, atol=1e-6)
