"""End-to-end sanity tests for the three TDSE solvers.

This is the pytest version of the original ``smoke.py`` script.  Each
solver gets its own focused test, the heavy free-Gaussian run is shared
via a session-scoped fixture, and quantitative tolerances are tightened
slightly relative to the original smoke pass/fail thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdse import (analytical, boundaries, grid as gridmod, potentials,
                  solvers, units)


# ---------------------------------------------------------------------------
# Shared, expensive setup: one free-Gaussian propagation per solver, reused
# across the assertion-style tests below.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def free_gaussian_setup():
    """Common grid / initial state / timestep for the leapfrog & CN runs."""
    g = gridmod.make_grid_nm(-50.0, 50.0, 801)
    sigma0 = 2.0 * units.NM
    x0 = -10.0 * units.NM
    E0 = 0.2 * units.EV
    k0 = units.wavenumber_from_energy(E0)

    psi0 = gridmod.gaussian_packet(g, x0, sigma0, k0)
    V = potentials.free(g)
    dt = gridmod.visscher_dt_max(g, V_max=0.0, safety=0.9)
    n_steps = 1200
    n_save = 60
    return {
        "grid": g,
        "psi0": psi0,
        "V": V,
        "dt": dt,
        "n_steps": n_steps,
        "n_save": n_save,
        "sigma0": sigma0,
        "x0": x0,
        "k0": k0,
    }


@pytest.fixture(scope="module")
def leapfrog_result(free_gaussian_setup):
    s = free_gaussian_setup
    bm = boundaries.hard_wall_mask(s["grid"])
    lf = solvers.LeapfrogSolver(s["grid"], s["V"], dt=s["dt"], bc="hard",
                                boundary_mask=bm)
    return lf.run(s["psi0"], n_steps=s["n_steps"], n_save=s["n_save"])


@pytest.fixture(scope="module")
def crank_nicolson_result(free_gaussian_setup):
    s = free_gaussian_setup
    bm = boundaries.hard_wall_mask(s["grid"])
    cn = solvers.CrankNicolsonSolver(s["grid"], s["V"], dt=s["dt"], bc="hard",
                                     boundary_mask=bm)
    return cn.run(s["psi0"], n_steps=s["n_steps"], n_save=s["n_save"])


@pytest.fixture(scope="module")
def split_operator_result(free_gaussian_setup):
    """Split-operator runs natively with periodic BC; the packet is far
    from the boundaries so this matches the hard-wall leapfrog/CN runs
    to the same accuracy the leapfrog<->CN test asserts."""
    s = free_gaussian_setup
    so = solvers.SplitOperatorSolver(s["grid"], s["V"], dt=s["dt"],
                                     bc="periodic")
    return so.run(s["psi0"], n_steps=s["n_steps"], n_save=s["n_save"])


# ---------------------------------------------------------------------------
# Visscher leapfrog
# ---------------------------------------------------------------------------

def test_leapfrog_visscher_norm_machine_precision(leapfrog_result):
    """The Visscher *modified* norm is conserved to machine precision."""
    nv = leapfrog_result.norm_visscher
    assert nv is not None, "leapfrog should expose the Visscher norm timeseries"
    drift = float(np.max(np.abs(nv - nv[0]) / nv[0]))
    assert drift < 1e-12, f"Visscher modified norm drifted by {drift:.2e}"


def test_leapfrog_ordinary_norm_bounded(leapfrog_result):
    """The ordinary norm wobbles at O(dt^2) but should not drift secularly."""
    n = leapfrog_result.norm
    drift = float(np.max(np.abs(n - n[0]) / n[0]))
    assert drift < 1e-3, f"ordinary norm drifted by {drift:.2e}"


def test_leapfrog_matches_analytic_free_gaussian(free_gaussian_setup,
                                                 leapfrog_result):
    """Numerical wave packet matches the closed-form free Gaussian.

    Leapfrog is O(dt^2) in time; over the 1200 steps used here the
    accumulated phase error gives an L2 disagreement around 10-20%.
    """
    s = free_gaussian_setup
    res = leapfrog_result
    i_check = len(res.times) // 2
    t_check = res.times[i_check]
    psi_num = res.psi[i_check]
    psi_ana = analytical.free_gaussian(
        s["grid"].x, t_check, s["x0"], s["sigma0"], s["k0"])
    err = np.linalg.norm(psi_num - psi_ana) / np.linalg.norm(psi_ana)
    assert err < 0.20, f"L2 relative error vs analytic: {err:.2e}"


# ---------------------------------------------------------------------------
# Crank-Nicolson reference solver
# ---------------------------------------------------------------------------

def test_crank_nicolson_norm_unitary(crank_nicolson_result):
    """CN with a sparse-LU solve should be numerically unitary."""
    n = crank_nicolson_result.norm
    drift = float(np.max(np.abs(n - n[0]) / n[0]))
    assert drift < 1e-10, f"CN norm drift: {drift:.2e}"


def test_crank_nicolson_matches_analytic_free_gaussian(free_gaussian_setup,
                                                       crank_nicolson_result):
    """CN should also match the closed-form free Gaussian to ~20%.

    CN is unconditionally stable and exactly unitary, but still O(dt^2)
    accurate -- so over 1200 steps the phase error sits at the same
    order as leapfrog.
    """
    s = free_gaussian_setup
    res = crank_nicolson_result
    i_check = len(res.times) // 2
    t_check = res.times[i_check]
    psi_num = res.psi[i_check]
    psi_ana = analytical.free_gaussian(
        s["grid"].x, t_check, s["x0"], s["sigma0"], s["k0"])
    err = np.linalg.norm(psi_num - psi_ana) / np.linalg.norm(psi_ana)
    assert err < 0.20, f"L2 relative error vs analytic: {err:.2e}"


# ---------------------------------------------------------------------------
# FTCS — must blow up
# ---------------------------------------------------------------------------

def test_ftcs_diverges(free_gaussian_setup):
    """FTCS is unconditionally unstable; the norm must explode."""
    s = free_gaussian_setup
    bm = boundaries.hard_wall_mask(s["grid"])
    ftcs = solvers.FTCSSolver(s["grid"], s["V"], dt=s["dt"], bc="hard",
                              boundary_mask=bm)
    res = ftcs.run(s["psi0"], n_steps=200, n_save=20)
    growth = res.norm[-1] / res.norm[0]
    assert growth > 100.0, (
        f"FTCS should blow up by orders of magnitude, but grew only "
        f"{growth:.2e} in 200 steps")


# ---------------------------------------------------------------------------
# Cross-solver agreement
# ---------------------------------------------------------------------------

def test_leapfrog_vs_crank_nicolson_agreement(leapfrog_result,
                                              crank_nicolson_result):
    """Leapfrog and CN should agree to a few percent at the halfway snapshot."""
    i = len(leapfrog_result.times) // 2
    psi_lf = leapfrog_result.psi[i]
    psi_cn = crank_nicolson_result.psi[i]
    err = np.linalg.norm(psi_lf - psi_cn) / np.linalg.norm(psi_cn)
    assert err < 0.05, f"leapfrog vs CN L2 disagreement: {err:.2e}"


# ---------------------------------------------------------------------------
# FFT split-operator solver
# ---------------------------------------------------------------------------

def test_split_operator_norm_unitary(split_operator_result):
    """Each Strang factor is exactly unitary; norm should drift only at FFT
    round-off level."""
    n = split_operator_result.norm
    drift = float(np.max(np.abs(n - n[0]) / n[0]))
    assert drift < 1e-10, f"split-operator norm drift: {drift:.2e}"


def test_split_operator_matches_analytic_free_gaussian(free_gaussian_setup,
                                                       split_operator_result):
    """Split-operator should match the closed-form free Gaussian.

    With FFT differentiation the kinetic-operator error is spectrally
    small; the only O(dt^2) error is the Strang time-splitting itself,
    so the agreement is at least as tight as leapfrog/CN.
    """
    s = free_gaussian_setup
    res = split_operator_result
    i_check = len(res.times) // 2
    t_check = res.times[i_check]
    psi_num = res.psi[i_check]
    psi_ana = analytical.free_gaussian(
        s["grid"].x, t_check, s["x0"], s["sigma0"], s["k0"])
    err = np.linalg.norm(psi_num - psi_ana) / np.linalg.norm(psi_ana)
    assert err < 0.20, f"L2 relative error vs analytic: {err:.2e}"


def test_split_operator_more_accurate_than_3point_stencil(free_gaussian_setup,
                                                          split_operator_result,
                                                          crank_nicolson_result):
    """Spectrally accurate kinetic operator => smaller error than CN/leapfrog.

    CN and leapfrog both use the 3-point Laplacian and so share an
    :math:`O(\\Delta x^2)` spatial-discretisation error.  The split-operator
    differentiates by FFT (spectrally accurate for a band-limited packet),
    so its remaining :math:`O(\\Delta t^2)` Strang error should be strictly
    smaller than the CN error against the analytic free-Gaussian.
    """
    s = free_gaussian_setup
    i = len(split_operator_result.times) // 2
    t_check = split_operator_result.times[i]
    psi_ana = analytical.free_gaussian(
        s["grid"].x, t_check, s["x0"], s["sigma0"], s["k0"])
    err_so = (np.linalg.norm(split_operator_result.psi[i] - psi_ana)
              / np.linalg.norm(psi_ana))
    err_cn = (np.linalg.norm(crank_nicolson_result.psi[i] - psi_ana)
              / np.linalg.norm(psi_ana))
    assert err_so < err_cn, (
        f"Split-operator error {err_so:.2e} should be smaller than "
        f"the 3-point stencil CN error {err_cn:.2e}")
