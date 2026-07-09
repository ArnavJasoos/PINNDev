"""Architecture matching phi(E).psi(A)."""

from __future__ import annotations

from pinnsystem.knowledge import match_architecture


def test_oscillatory_multiscale_prefers_spectral_arch():
    out = match_architecture(
        {"periodicity": 0.9, "geometry_complexity": 0.1, "multiscale": 0.9}
    )
    assert out["architecture"] in {"Fourier-MLP", "SIREN"}
    assert out["scores"]


def test_empty_features_defaults_to_mlp():
    out = match_architecture({})
    assert out["architecture"] == "MLP"


def test_forbidden_architecture_is_excluded():
    fv = {"periodicity": 0.9, "geometry_complexity": 0.1, "multiscale": 0.9}
    first = match_architecture(fv)["architecture"]
    out = match_architecture(fv, forbidden=[first])
    assert out["architecture"] != first
