"""Cheap, sub-second sanity tests that don't run any time-stepping.

These exist so that ``pytest -m "not slow"`` plus ``test_solvers.py`` is
genuinely fast (few seconds total), giving developers a quick green/red
signal without waiting for the full §3-§9 physics suite.
"""

from __future__ import annotations

import numpy as np

from tdse import (analytical, grid as gridmod, observables, potentials,
                  solvers, units)


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

def test_isw_eigenstates_orthonormal():
    """ISW eigenstates should be L^2-orthonormal on the grid."""
    g = gridmod.make_grid_nm(-2.0, 2.0, 401)
    L_left, L_right = -1.0 * units.NM, +1.0 * units.NM
    phi = np.array([analytical.isw_eigenstate(n, g.x, L_left, L_right)
                    for n in range(1, 5)])
    # diagonal: norms ~ 1
    norms = np.sum(np.abs(phi) ** 2, axis=-1) * g.dx
    assert np.allclose(norms, 1.0, atol=5e-3)
    # off-diagonal: orthogonality
    for i in range(4):
        for j in range(i + 1, 4):
            ovl = np.sum(phi[i] * phi[j]) * g.dx
            assert abs(ovl) < 1e-2, f"<phi_{i+1}|phi_{j+1}> = {ovl:.2e}"


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


# ---------------------------------------------------------------------------
# Imaginary-time relaxation: ground-state energy of an SHO
# ---------------------------------------------------------------------------

def test_imaginary_time_ground_state_sho():
    """Imag-time relaxation should converge to E_0 = (1/2) hbar omega."""
    omega = 2.0 * np.pi * 5e13
    g = gridmod.make_grid_nm(-3.0, 3.0, 301)
    V = 0.5 * units.M_E * omega ** 2 * g.x ** 2
    _psi, E0, _hist = solvers.ground_state_imag_time(g, V, n_steps=1500)
    E_exact = 0.5 * units.HBAR * omega
    rel_err = abs(E0 - E_exact) / E_exact
    assert rel_err < 5e-3, (
        f"imag-time E_0 = {E0 / units.EV * 1e3:.4f} meV, "
        f"exact = {E_exact / units.EV * 1e3:.4f} meV "
        f"(rel err {rel_err:.2e})")


def test_potential_factory_static_vs_time_dependent_flag():
    g = gridmod.make_grid_nm(-2.0, 2.0, 51)
    V_static = potentials.harmonic(g, omega=1e15)
    assert V_static.is_time_dependent is False
    assert isinstance(V_static(g.x, 0.0), np.ndarray)

    V_td = potentials.driven_harmonic(omega=1e15, F0=1e-12, omega_drive=1e15)
    assert V_td.is_time_dependent is True
    assert isinstance(V_td(g.x, 0.0), np.ndarray)
    assert V_td(g.x, 0.0).shape == g.x.shape
