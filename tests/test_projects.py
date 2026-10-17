"""Physics checks for the three simulations in the final project.

Each test re-runs a (smaller or identical) version of a notebook simulation and
compares it with an exact result. They take ~30 s in total, so they are marked
``slow`` (CI runs them in a separate step).
"""

from __future__ import annotations

import numpy as np
import pytest

from tdse import (analytical, boundaries, grid as gridmod, observables, potentials,
                  solvers, units)

NM, FS, EV = units.NM, units.FS, units.EV
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Part I: the leapfrog stability limit is sharp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factor, stable", [(0.95, True), (1.10, False)])
def test_leapfrog_stability_limit_is_sharp(factor, stable):
    g = gridmod.make_grid_nm(-15.0, 15.0, 301)
    psi0 = gridmod.gaussian_packet(g, 0.0, 1.0 * NM, units.wavenumber_from_energy(0.3 * EV))
    dt = factor * gridmod.visscher_dt_max(g, V_max=0.0, safety=1.0)
    r = solvers.LeapfrogSolver(g, potentials.free(g), dt=dt, bc="hard").run(
        psi0, n_steps=int(30 * FS / dt), n_save=10)
    growth = np.nanmax(r.norm) / r.norm[0]
    if stable:
        assert growth < 1 + 1e-4          # ordinary norm only wobbles at O(dt^2)
    else:
        assert not np.isfinite(r.norm[-1]) or growth > 1e10


# ---------------------------------------------------------------------------
# Part II.1: free particle
# ---------------------------------------------------------------------------

def _free_run(n_points):
    sigma0, x0 = 2.0 * NM, -20.0 * NM
    k0 = units.wavenumber_from_energy(0.5 * EV)
    g = gridmod.make_grid_nm(-60, 60, n_points)
    dt = gridmod.visscher_dt_max(g, V_max=0.0, safety=0.5)
    res = solvers.LeapfrogSolver(g, potentials.free(g), dt=dt, bc="hard").run(
        gridmod.gaussian_packet(g, x0, sigma0, k0), n_steps=int(80 * FS / dt), n_save=40)
    ts = observables.timeseries(res)
    width = np.sqrt(ts["x_var"])
    exact = np.array([analytical.free_gaussian_width(t, sigma0) for t in ts["t"]])
    return g, k0, ts, np.max(np.abs(width / exact - 1))


def test_free_particle_moves_at_the_grid_group_velocity():
    """On a grid the group velocity is (hbar/m) sin(k dx)/dx, not hbar k/m."""
    g, k0, ts, _ = _free_run(1201)
    v_measured = np.polyfit(ts["t"], ts["x_mean"], 1)[0]
    v_grid = units.HBAR / units.M_E * np.sin(k0 * g.dx) / g.dx
    assert abs(v_measured / v_grid - 1) < 2e-3


def test_free_particle_spreading_converges_to_exact():
    """Halving dx cuts the width error about 4x (second-order scheme)."""
    *_, err_coarse = _free_run(1201)
    *_, err_fine = _free_run(2401)
    assert err_fine < 0.015
    assert err_coarse / err_fine > 3.0


# ---------------------------------------------------------------------------
# Part II.3: superposition (harmonic-oscillator coherent state)
# ---------------------------------------------------------------------------

def test_coherent_state_moves_classically_without_spreading():
    omega = 2 * np.pi * 3.7e13
    sigma = np.sqrt(units.HBAR / (2 * units.M_E * omega))
    x_cl = 1.7642 * NM                       # amplitude used in the notebook
    alpha = x_cl / (2 * sigma)               # <x> = 2 sigma alpha for a coherent state
    period = 2 * np.pi / omega
    g = gridmod.make_grid_nm(-5, 5, 801)
    V = potentials.harmonic(g, omega)
    dt = gridmod.visscher_dt_max(g, V_max=V.max_abs(g.x), safety=0.5)
    res = solvers.LeapfrogSolver(g, V, dt=dt, bc="hard").run(
        gridmod.gaussian_packet(g, x_cl, sigma, 0.0), n_steps=int(2 * period / dt), n_save=60)
    ts = observables.timeseries(res, V_static=V.V)
    assert np.max(np.abs(ts["x_mean"] - x_cl * np.cos(omega * ts["t"]))) < 0.01 * x_cl
    width = np.sqrt(ts["x_var"])
    assert np.ptp(width) / width.mean() < 0.01
    expected_energy = units.HBAR * omega * (alpha ** 2 + 0.5)
    assert abs(ts["E_mean"].mean() / expected_energy - 1) < 0.01


# ---------------------------------------------------------------------------
# Part III: tunnelling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("E_eV", [0.9, 3.0])
def test_tunnelling_matches_transfer_matrix(E_eV):
    V0, a, sigma = 2.0 * EV, 0.6 * NM, 1.5 * NM
    g = gridmod.make_grid_nm(-50, 50, 4001)
    V = potentials.rectangular_barrier(g, -a / 2, a / 2, V0)
    bm = boundaries.absorbing_mask(g, sponge_width=6.0 * NM)
    dt = gridmod.visscher_dt_max(g, V_max=V0 + np.max(np.abs(bm.V_imag)), safety=0.5)
    E = E_eV * EV
    psi0 = gridmod.gaussian_packet(g, -15 * NM, sigma, units.wavenumber_from_energy(E))
    res = solvers.LeapfrogSolver(g, V, dt=dt, bc="absorbing", boundary_mask=bm).run(
        psi0, n_steps=int(150 * FS / dt), n_save=1500, track_energy=False)
    T_sim = observables.transmitted_probability(res, 5 * NM)

    # exact T(E) averaged over the packet's spread of energies
    g_tm = gridmod.make_grid_nm(-5, 5, 2001)
    V_tm = potentials.rectangular_barrier(g_tm, -a / 2, a / 2, V0)
    k0 = units.wavenumber_from_energy(E)
    k = np.linspace(k0 - 2 / sigma, k0 + 2 / sigma, 81)
    k = k[k > 0]
    w = np.exp(-2 * ((k - k0) * sigma) ** 2)
    T_k = [analytical.transfer_matrix_TR((units.HBAR * kk) ** 2 / (2 * units.M_E), g_tm.x, V_tm.V)[0]
           for kk in k]
    T_exact = float(np.sum(w * np.array(T_k)) / np.sum(w))
    assert abs(T_sim / T_exact - 1) < 0.03, f"T_sim = {T_sim:.4g}, T_exact = {T_exact:.4g}"
