"""Tests for the §3-§9 time-dependent physics experiments.

This is the pytest version of the original ``smoke_03.py`` script.  Each
section is a separate test, marked ``slow`` so the suite can be run as
either ``pytest`` (everything) or ``pytest -m "not slow"`` (the fast
solver tests in ``test_solvers.py`` only).
"""

from __future__ import annotations

import numpy as np
import pytest

from tdse import (analytical, boundaries, grid as gridmod, observables,
                  potentials, solvers, units)


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# §3  Rabi oscillations in a driven infinite square well
# ---------------------------------------------------------------------------

def test_rabi_oscillation_in_driven_isw():
    L = 2.0 * units.NM
    g = gridmod.make_grid_nm(-L / (2 * units.NM), L / (2 * units.NM), 81)
    xL_w, xR_w = -L / 2, +L / 2
    phi1 = analytical.isw_eigenstate(1, g.x, xL_w, xR_w)
    phi2 = analytical.isw_eigenstate(2, g.x, xL_w, xR_w)
    E1 = analytical.isw_energy(1, xL_w, xR_w)
    E2 = analytical.isw_energy(2, xL_w, xR_w)
    omega_12 = (E2 - E1) / units.HBAR
    F0 = 0.15 * units.EV / units.NM
    x12 = -16.0 * L / (9.0 * np.pi ** 2)
    Omega_R = F0 * abs(x12) / units.HBAR
    T_R = 2 * np.pi / Omega_R

    V = potentials.pulsed_dipole(potentials.free(g).V, F0=F0,
                                 omega_drive=omega_12,
                                 t_centre=0.0, t_width=1e30,
                                 envelope="gaussian")
    dt = gridmod.visscher_dt_max(g, V_max=F0 * (L / 2), safety=0.4)
    n_steps = int(0.6 * T_R / dt)
    lf = solvers.LeapfrogSolver(g, V, dt=dt, bc="hard")
    res = lf.run(phi1.astype(complex), n_steps=n_steps, n_save=80)
    P = observables.populations(res, np.array([phi1, phi2]))
    P2_max = float(np.max(P[:, 1]))
    assert P2_max > 0.3, f"no visible Rabi flip: peak P2 = {P2_max:.3f}"


# ---------------------------------------------------------------------------
# §4  Landau-Zener: monotonicity of P_diabatic vs sweep rate
# ---------------------------------------------------------------------------

def _run_LZ(g, xL, xR, sigma_w, d0, alpha, t_max):
    V_lz = potentials.landau_zener_wells(g, x_L=xL, x_R=xR, sigma=sigma_w,
                                         depth0=d0, sweep_rate=alpha)
    psi_init, _, _ = solvers.ground_state_imag_time(
        g, V_lz(g.x, -t_max), n_steps=800)
    psi_final_gs, _, _ = solvers.ground_state_imag_time(
        g, V_lz(g.x, +t_max), n_steps=800)
    dt = gridmod.visscher_dt_max(g, V_max=d0 + 0.1 * units.EV, safety=0.4)
    lf = solvers.LeapfrogSolver(g, V_lz, dt=dt, bc="hard")
    n_steps = int(2 * t_max / dt)
    res = lf.run(psi_init.astype(complex), n_steps=n_steps, n_save=10)
    ovl = np.sum(np.conj(psi_final_gs) * res.psi[-1]) * g.dx
    return 1.0 - abs(ovl) ** 2


def test_landau_zener_fast_vs_slow_sweep():
    xL, xR = -1.0 * units.NM, +1.0 * units.NM
    sigma_w = 0.50 * units.NM
    d0 = 0.50 * units.EV
    g = gridmod.make_grid_nm(-8, 8, 201)

    V_static = -d0 * (np.exp(-((g.x - xL) ** 2) / (2 * sigma_w ** 2))
                      + np.exp(-((g.x - xR) ** 2) / (2 * sigma_w ** 2)))
    Es, _ = solvers.bound_eigenstates(g, V_static, n_states=4)
    Delta = 0.5 * (Es[1] - Es[0])
    assert Delta > 1e-3 * units.EV, \
        f"degenerate-well minimum gap too small: {Delta / units.EV * 1e3:.3f} meV"

    P_fast = _run_LZ(g, xL, xR, sigma_w, d0,
                     2e-1 * units.EV / units.FS, 30 * units.FS)
    alpha_slow = 5e-4 * units.EV / units.FS
    P_slow = _run_LZ(g, xL, xR, sigma_w, d0, alpha_slow,
                     max(30 * units.FS, 20 * Delta / alpha_slow))
    assert P_fast > 0.9, f"fast sweep should be ~diabatic: P={P_fast:.3f}"
    assert P_slow < P_fast - 0.1, (
        f"expected adiabatic suppression: P_fast={P_fast:.3f}, "
        f"P_slow={P_slow:.3f}")


# ---------------------------------------------------------------------------
# §5  Adiabatic vs sudden: a very fast switch matches the sudden overlap
# ---------------------------------------------------------------------------

def test_widening_well_sudden_limit():
    V_wall = 1.5 * units.EV
    L0, L1 = 1.5 * units.NM, 3.0 * units.NM
    g = gridmod.make_grid_nm(-3.0, 3.0, 151)
    V_init = np.where(np.abs(g.x) <= L0 / 2, 0.0, V_wall)
    Ei, phis_i = solvers.bound_eigenstates(g, V_init, n_states=3)
    tau_ad = units.HBAR / (Ei[1] - Ei[0])
    tau = 0.05 * tau_ad

    V_ww = potentials.widening_well(g, L0=L0, L1=L1,
                                    t_switch=3 * tau, t_smooth=tau,
                                    V_wall=V_wall)
    dt = gridmod.visscher_dt_max(g, V_max=V_wall, safety=0.4)
    T_total = 3 * tau + 15 * units.FS
    n_steps = int(T_total / dt)
    lf = solvers.LeapfrogSolver(g, V_ww, dt=dt, bc="hard")
    res = lf.run(phis_i[0].astype(complex), n_steps=n_steps, n_save=30)

    V_final = np.where(np.abs(g.x) <= L1 / 2, 0.0, V_wall)
    _, phis_f = solvers.bound_eigenstates(g, V_final, n_states=5)
    P_final = observables.populations(res, phis_f)[-1]
    P_sudden = np.array([abs(np.sum(np.conj(p) * phis_i[0]) * g.dx) ** 2
                         for p in phis_f])
    assert abs(P_final[0] - P_sudden[0]) < 0.1, (
        f"sudden limit P_0 mismatch: measured={P_final[0]:.3f}, "
        f"sudden={P_sudden[0]:.3f}")


# ---------------------------------------------------------------------------
# §6  Wave-packet revivals in the infinite well
# ---------------------------------------------------------------------------

def test_isw_half_revival_mirrors_initial_packet():
    L = 4.0 * units.NM
    g = gridmod.make_grid_nm(-L / (2 * units.NM), L / (2 * units.NM), 201)
    sigma0 = 0.25 * units.NM
    psi0 = gridmod.gaussian_packet(g, x0=-L / 4, sigma=sigma0, k0=0.0)
    E1 = analytical.isw_energy(1, -L / 2, +L / 2)
    T_rev = 2 * np.pi * units.HBAR / E1
    dt = 0.2 * units.FS
    n_steps = int(0.5 * T_rev / dt)
    cn = solvers.CrankNicolsonSolver(g, potentials.free(g), dt=dt, bc="hard")
    res = cn.run(psi0, n_steps=n_steps, n_save=80)
    rho = np.abs(res.psi[-1]) ** 2
    x_peak = g.x[int(np.argmax(rho))]
    assert x_peak > 0, (
        f"at T_rev/2 the packet should peak at +x (mirror of -L/4); "
        f"got x_peak = {x_peak * 1e9:.3f} nm")


# ---------------------------------------------------------------------------
# §7  Autocorrelation spectroscopy of an SHO coherent state
# ---------------------------------------------------------------------------

def test_sho_autocorrelation_recovers_eigenvalues():
    omega0 = 2 * np.pi * 3.7e13
    sigma0 = np.sqrt(units.HBAR / (2 * units.M_E * omega0))
    g = gridmod.make_grid_nm(-4, 4, 151)
    V_sho = potentials.harmonic(g, omega=omega0)
    psi0 = gridmod.gaussian_packet(g, x0=2 * sigma0, sigma=sigma0, k0=0.0)
    dt = gridmod.visscher_dt_max(g, V_max=V_sho.max_abs(g.x), safety=0.5)
    T_sho = 2 * np.pi / omega0
    n_periods = 8
    n_steps = int(n_periods * T_sho / dt)
    lf = solvers.LeapfrogSolver(g, V_sho, dt=dt, bc="hard")
    res = lf.run(psi0.astype(complex), n_steps=n_steps, n_save=n_steps)

    C = observables.autocorrelation(res, psi0.astype(complex))
    dt_sav = res.times[1] - res.times[0]
    omega, S = observables.power_spectrum(C, dt_sav, window="hann")
    omega_abs = np.abs(omega)
    peak_idx = int(np.argmax(S))
    peak_E_eV = units.HBAR * omega_abs[peak_idx] / units.EV

    hw = units.HBAR * omega0 / units.EV
    deltas = [abs(peak_E_eV - (n + 0.5) * hw) for n in range(6)]
    assert min(deltas) < 0.15 * hw, (
        f"brightest spectral peak {peak_E_eV * 1e3:.2f} meV does not match "
        f"any (n+1/2) hbar*omega in [0..5]; closest miss = "
        f"{min(deltas) * 1e3:.2f} meV")


# ---------------------------------------------------------------------------
# §8  Photoionisation of a 1D finite well
# ---------------------------------------------------------------------------

def test_photoionisation_drains_bound_state():
    g = gridmod.make_grid_nm(-15, 15, 201)
    V_well = potentials.finite_square_well(g, -0.5 * units.NM, 0.5 * units.NM,
                                           depth=2.5 * units.EV).V
    psi_gs, E_gs, _ = solvers.ground_state_imag_time(g, V_well, n_steps=1500)
    _, phib = solvers.bound_eigenstates(g, V_well, n_states=2)
    omega_pulse = 2.5 * abs(E_gs) / units.HBAR
    F0 = 0.05 * units.EV / units.NM
    V_pulse = potentials.pulsed_dipole(V_well, F0=F0, omega_drive=omega_pulse,
                                       t_centre=5 * units.FS,
                                       t_width=1.5 * units.FS,
                                       envelope="gaussian")
    bm = boundaries.absorbing_mask(g, 3 * units.NM)
    V_max = max(2.5 * units.EV, F0 * (g.L / 2)) + np.max(np.abs(bm.V_imag))
    dt = gridmod.visscher_dt_max(g, V_max=V_max, safety=0.4)
    n_steps = int(15 * units.FS / dt)
    lf = solvers.LeapfrogSolver(g, V_pulse, dt=dt, bc="absorbing",
                                boundary_mask=bm)
    res = lf.run(psi_gs.astype(complex), n_steps=n_steps, n_save=30)
    P_bound = observables.populations(res, phib[:1])[:, 0]
    P_drop = P_bound[0] - P_bound[-1]
    assert P_drop > 1e-3, (
        f"expected visible bound-state drain; got P_drop = {P_drop:.3e}")


# ---------------------------------------------------------------------------
# §9  HHG mini-run: dipole-acceleration spectrum is non-empty
# ---------------------------------------------------------------------------

def test_hhg_dipole_acceleration_spectrum_nonempty():
    a_soft = 0.15 * units.NM
    g = gridmod.make_grid_nm(-8, 8, 251)
    V_sc = potentials.soft_coulomb(g, Z=1.0, a_soft=a_soft)
    psi_gs, E_gs, _ = solvers.ground_state_imag_time(g, V_sc.V, n_steps=1500)
    Ip = -E_gs
    hbar_Omega = 0.6 * units.EV
    omega_L = hbar_Omega / units.HBAR
    T_cycle = 2 * np.pi / omega_L
    n_cycles = 3
    T_p = n_cycles * T_cycle
    U_target = 0.5 * Ip
    E_field_amp = (np.sqrt(4 * units.M_E * omega_L ** 2 * U_target)
                   / units.E_CHARGE)
    F0_hhg = units.E_CHARGE * E_field_amp
    V_hhg = potentials.pulsed_dipole(V_sc.V, F0=F0_hhg, omega_drive=omega_L,
                                     t_centre=T_p / 2, t_width=0.0,
                                     envelope="sin2", n_cycles=n_cycles)
    bm = boundaries.absorbing_mask(g, 2 * units.NM)
    V_max = (max(abs(np.min(V_sc.V)), F0_hhg * (g.L / 2))
             + np.max(np.abs(bm.V_imag)))
    dt = gridmod.visscher_dt_max(g, V_max=V_max, safety=0.35)
    n_steps = int(T_p / dt)
    lf = solvers.LeapfrogSolver(g, V_hhg, dt=dt, bc="absorbing",
                                boundary_mask=bm)
    res = lf.run(psi_gs.astype(complex), n_steps=n_steps,
                 n_save=min(n_steps + 1, 400))

    dVdx = np.gradient(V_sc.V, g.dx)

    def field_fn(t):
        if t <= 0 or t >= T_p:
            return 0.0
        env = np.sin(np.pi * t / T_p) ** 2
        return (F0_hhg / units.E_CHARGE) * env * np.cos(omega_L * t)

    a_t = observables.dipole_acceleration(res, dVdx, field_fn=field_fn,
                                          q=-units.E_CHARGE)
    dt_sav = res.times[1] - res.times[0]
    _omega, S = observables.power_spectrum(a_t, dt_sav, window="hann")
    peak_S = float(np.max(S[1:]))
    assert peak_S > 0, "HHG spectrum is empty"
    assert np.max(np.abs(a_t)) > 0, "dipole acceleration is identically zero"
