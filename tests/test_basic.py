"""Cheap, sub-second sanity tests that don't run any time-stepping.

These exist so that ``pytest -m "not slow"`` plus ``test_solvers.py`` is
genuinely fast (few seconds total), giving developers a quick green/red
signal; the physics of the three simulations is in ``test_projects.py``.
"""

from __future__ import annotations

import numpy as np

from tdse import analytical, grid as gridmod, observables, potentials, units


# ---------------------------------------------------------------------------
# Package import + module surface
# ---------------------------------------------------------------------------

def test_package_imports():
    """Every documented submodule must be importable from ``tdse``."""
    import tdse  # noqa: F401  (import-time side effects)
    for name in ["units", "grid", "potentials", "boundaries", "solvers",
                 "observables", "analytical", "viz"]:
        assert hasattr(tdse, name), f"tdse.{name} missing"


# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018, used throughout)
# ---------------------------------------------------------------------------

def test_si_constants_have_expected_values():
    assert units.HBAR == 1.054_571_817e-34
    assert units.M_E == 9.109_383_701_5e-31
    assert units.E_CHARGE == 1.602_176_634e-19
    assert units.EV == units.E_CHARGE


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def test_grid_dx_and_endpoints():
    g = gridmod.make_grid_nm(-10.0, 10.0, 201)
    assert g.N == 201
    assert g.x.shape == (201,)
    assert g.x[0] == -10.0 * units.NM
    assert g.x[-1] == +10.0 * units.NM
    expected_dx = (g.x[-1] - g.x[0]) / (g.N - 1)
    assert abs(g.dx - expected_dx) < 1e-30


def test_gaussian_packet_is_normalised():
    g = gridmod.make_grid_nm(-20.0, 20.0, 401)
    psi = gridmod.gaussian_packet(g, 0.0, 1.0 * units.NM,
                                  k0=units.wavenumber_from_energy(0.5 * units.EV))
    norm = float(np.sum(np.abs(psi) ** 2) * g.dx)
    assert abs(norm - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Analytical references
# ---------------------------------------------------------------------------

def test_barrier_transmission_matches_transfer_matrix():
    """Closed-form rectangular-barrier T(E) ≈ transfer-matrix T(E)."""
    g = gridmod.make_grid_nm(-5.0, 5.0, 1001)
    V0 = 1.0 * units.EV
    a = 0.5 * units.NM
    V = np.where((g.x >= -a / 2) & (g.x <= +a / 2), V0, 0.0)
    for E_eV in [0.3, 0.5, 0.8, 1.5, 2.5]:
        E = E_eV * units.EV
        T_ana = analytical.barrier_transmission(E, V0, a)
        T_tm, _R_tm = analytical.transfer_matrix_TR(E, g.x, V)
        assert abs(T_ana - T_tm) < 0.02, (
            f"E = {E_eV:.2f} eV: analytic T = {T_ana:.4f}, "
            f"transfer matrix T = {T_tm:.4f}")


# ---------------------------------------------------------------------------
# Observables on a known state
# ---------------------------------------------------------------------------

def test_momentum_expectation_matches_k0_for_gaussian():
    """<p> of a Gaussian packet with central wavenumber k0 should be hbar k0."""
    g = gridmod.make_grid_nm(-30.0, 30.0, 1201)
    k0 = units.wavenumber_from_energy(0.4 * units.EV)
    psi = gridmod.gaussian_packet(g, 0.0, 2.0 * units.NM, k0)
    p_meas = observables.momentum_expectation(psi, g.dx)
    p_expected = units.HBAR * k0
    rel_err = abs(p_meas - p_expected) / p_expected
    # Central-difference truncation error is ~ (k0 * dx)^2 / 6 ~= 4e-3 here.
    assert rel_err < 1e-2, f"<p> error = {rel_err:.2e}"


def test_position_expectation_at_centre():
    g = gridmod.make_grid_nm(-30.0, 30.0, 601)
    x0 = 3.0 * units.NM
    psi = gridmod.gaussian_packet(g, x0, 1.0 * units.NM, 0.0)
    x_meas = observables.position_expectation(psi, g.x, g.dx)
    rel_err = abs(x_meas - x0) / abs(x0)
    assert rel_err < 1e-3, f"<x> = {x_meas * 1e9:.4f} nm, expected {x0 * 1e9:.4f} nm"


def test_potentials_used_in_the_project():
    g = gridmod.make_grid_nm(-2.0, 2.0, 51)
    assert np.all(potentials.free(g)(g.x) == 0.0)
    V_sho = potentials.harmonic(g, omega=1e15)
    assert V_sho.is_time_dependent is False
    assert np.argmin(V_sho(g.x)) == g.N // 2
    V_bar = potentials.rectangular_barrier(g, -0.3 * units.NM, 0.3 * units.NM, 2.0 * units.EV)
    inside = np.abs(g.x) <= 0.3 * units.NM
    assert np.allclose(V_bar(g.x)[inside], 2.0 * units.EV)
    assert np.allclose(V_bar(g.x)[~inside], 0.0)


def test_coherent_state_is_a_superposition_of_eigenstates():
    """A displaced ground-state Gaussian = coherent state: its overlaps with the
    oscillator eigenstates follow a Poisson distribution with mean |alpha|^2."""
    omega = 2 * np.pi * 3.7e13
    sigma = np.sqrt(units.HBAR / (2 * units.M_E * omega))
    alpha = 1.77
    g = gridmod.make_grid_nm(-6.0, 6.0, 1201)
    psi = gridmod.gaussian_packet(g, 2 * alpha * sigma, sigma, 0.0)   # <x> = 2 sigma alpha
    probs = []
    for n in range(20):
        phi = analytical.sho_eigenstate(n, g.x, omega)
        phi = phi / np.sqrt(np.sum(np.abs(phi) ** 2) * g.dx)
        probs.append(abs(np.sum(np.conj(phi) * psi) * g.dx) ** 2)
    probs = np.array(probs)
    poisson = np.exp(-alpha ** 2) * alpha ** (2 * np.arange(20)) / np.cumprod(np.r_[1, np.arange(1, 20)])
    assert abs(probs.sum() - 1) < 1e-3
    assert np.max(np.abs(probs - poisson)) < 1e-3
