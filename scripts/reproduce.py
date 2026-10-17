"""Re-run every result in the final project and redraw its figures.

Part I   - stress tests: which of the four solvers can be trusted?
Part II  - free particle, and a superposition (harmonic-oscillator coherent state)
Part III - quantum tunnelling through a rectangular barrier

Usage:  python scripts/reproduce.py           (figures/*.png and results/summary.json, ~1 min)
        python scripts/reproduce.py --gifs    (also redraw the three animated GIFs)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tdse import analytical, boundaries, grid as gridmod, observables, potentials, solvers, units, viz

ROOT = Path(__file__).resolve().parent.parent
FIG, RES = ROOT / "figures", ROOT / "results"
FS, NM, EV = units.FS, units.NM, units.EV

# Chart colours (checked for colour-vision deficiency); text stays in neutral ink.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 150, "lines.linewidth": 2,
    "legend.frameon": False, "axes.titlesize": 10.5, "axes.titlelocation": "left",
})


def rel_drift(series: np.ndarray) -> float:
    return float(np.max(np.abs(series - series[0])) / abs(series[0]))


# ----- Part I: stress tests ---------------------------------------------------------------------

def part1(summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # I.1 Forward Euler (FTCS): unconditionally unstable
    g = gridmod.make_grid_nm(-15.0, 15.0, 301)
    psi0 = gridmod.gaussian_packet(g, 0.0, 1.0 * NM, units.wavenumber_from_energy(0.5 * EV))
    dt = gridmod.visscher_dt_max(g, V_max=0.0, safety=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ftcs = solvers.FTCSSolver(g, potentials.free(g), dt=dt).run(psi0, n_steps=400, n_save=20)
    finite = np.isfinite(ftcs.norm) & (ftcs.norm > 0)
    decades = float(np.log10(ftcs.norm[finite][-1] / ftcs.norm[0]))
    summary["ftcs_norm_growth_decades_400_steps"] = round(decades, 1)
    ax = axes[0]
    ax.semilogy(ftcs.diag_times[finite] / FS, ftcs.norm[finite], color=ORANGE)
    ax.set(xlabel="t (fs)", ylabel="norm  ∫|ψ|² dx  (log)",
           title=f"I.1  Forward Euler blows up: norm grows {decades:.0f} decades in 400 steps")

    # I.2 / I.5 Norm and energy conservation: leapfrog vs Crank-Nicolson vs FFT split-operator
    g = gridmod.make_grid_nm(-30.0, 30.0, 601)
    psi0 = gridmod.gaussian_packet(g, 0.0, 1.5 * NM, units.wavenumber_from_energy(0.4 * EV))
    dt = gridmod.visscher_dt_max(g, V_max=0.0, safety=0.9)
    V = potentials.free(g)
    lf = solvers.LeapfrogSolver(g, V, dt=dt, bc="hard").run(psi0, n_steps=600, n_save=60)
    cn = solvers.CrankNicolsonSolver(g, V, dt=dt, bc="hard").run(psi0, n_steps=600, n_save=60)
    so = solvers.SplitOperatorSolver(g, V, dt=dt, bc="periodic").run(psi0, n_steps=600, n_save=60)
    summary["norm_drift"] = {
        "leapfrog_visscher_norm": rel_drift(lf.norm_visscher),
        "leapfrog_ordinary_norm": rel_drift(lf.norm),
        "crank_nicolson": rel_drift(cn.norm),
        "split_operator": rel_drift(so.norm),
    }
    summary["energy_drift"] = {name: rel_drift(r.energy) for name, r in
                               [("leapfrog", lf), ("crank_nicolson", cn), ("split_operator", so)]}
    ax = axes[1]
    t = lf.diag_times / FS
    eps = 1e-17
    ax.semilogy(t, np.abs(lf.norm_visscher - lf.norm_visscher[0]) + eps, color=BLUE,
                label="leapfrog, Visscher norm")
    ax.semilogy(t, np.abs(lf.norm - lf.norm[0]) + eps, color=YELLOW, label="leapfrog, ordinary norm")
    ax.semilogy(cn.diag_times / FS, np.abs(cn.norm - cn.norm[0]) + eps, color=AQUA,
                label="Crank–Nicolson")
    ax.semilogy(so.diag_times / FS, np.abs(so.norm - so.norm[0]) + eps, color=ORANGE,
                label="FFT split-operator")
    ax.set(xlabel="t (fs)", ylabel="|norm(t) − norm(0)|  (log)", ylim=(1e-17, 1e-5),
           title="I.2  Probability conservation, free Gaussian")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    # I.3 Is the stability (CFL) bound sharp?
    g = gridmod.make_grid_nm(-15.0, 15.0, 301)
    psi0 = gridmod.gaussian_packet(g, 0.0, 1.0 * NM, units.wavenumber_from_energy(0.3 * EV))
    dt_crit = gridmod.visscher_dt_max(g, V_max=0.0, safety=1.0)
    growth = {}
    ax = axes[2]
    for factor, color, style in [(0.50, BLUE, "-"), (0.95, AQUA, "-"), (1.02, YELLOW, "--"),
                                 (1.10, ORANGE, ":")]:
        dt = factor * dt_crit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = solvers.LeapfrogSolver(g, potentials.free(g), dt=dt, bc="hard").run(
                psi0, n_steps=int(60 * FS / dt), n_save=20)
        ok = np.isfinite(r.norm)
        growth[str(factor)] = float(np.log10(np.max(r.norm[ok]) / r.norm[0]))
        ax.semilogy(r.diag_times[ok] / FS, r.norm[ok], color=color, linestyle=style,
                    label=f"{factor:.2f} × Δt_crit")
    summary["cfl_norm_growth_decades_60fs"] = {k: round(v, 2) for k, v in growth.items()}
    ax.set(xlabel="t (fs)", ylabel="norm  (log)", ylim=(0.3, 1e15),
           title="I.3  The leapfrog stability limit is a cliff, not a slope")
    ax.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    fig.savefig(FIG / "part1_solver_tests.png")
    plt.close(fig)


# ----- Part II: free particle and superposition ---------------------------------------------------

def run_free(n_points: int, sigma0: float, x0: float, k0: float):
    g = gridmod.make_grid_nm(-60, 60, n_points)
    V = potentials.free(g)
    dt = gridmod.visscher_dt_max(g, V_max=0.0, safety=0.5)
    res = solvers.LeapfrogSolver(g, V, dt=dt, bc="hard").run(
        gridmod.gaussian_packet(g, x0, sigma0, k0), n_steps=int(80 * FS / dt), n_save=160)
    ts = observables.timeseries(res, V_static=V.V)
    width = np.sqrt(ts["x_var"])
    width_exact = np.array([analytical.free_gaussian_width(tt, sigma0) for tt in ts["t"]])
    return g, V, res, ts, width, width_exact


def free_particle(summary: dict, gifs: bool) -> None:
    sigma0, x0, E0 = 2.0 * NM, -20.0 * NM, 0.5 * EV
    k0 = units.wavenumber_from_energy(E0)
    g, V, res, ts, width, width_exact = run_free(1201, sigma0, x0, k0)
    t = ts["t"]
    v_g = units.HBAR * k0 / units.M_E
    # On a grid with spacing dx the kinetic energy is 2T(1 - cos k dx), so waves travel at
    # (hbar/m) sin(k dx)/dx instead of hbar k/m: a known, predictable grid error.
    v_grid = units.HBAR / units.M_E * np.sin(k0 * g.dx) / g.dx
    *_, width_fine, width_exact_fine = run_free(2401, sigma0, x0, k0)
    summary["free_particle"] = {
        "dx_nm": g.dx / NM,
        "group_velocity_nm_per_fs_exact": v_g / (NM / FS),
        "group_velocity_nm_per_fs_on_grid_theory": v_grid / (NM / FS),
        "group_velocity_nm_per_fs_measured": float(np.polyfit(t, ts["x_mean"], 1)[0] / (NM / FS)),
        "width_growth_measured": float(width[-1] / width[0]),
        "width_growth_exact": float(width_exact[-1] / width_exact[0]),
        "max_width_error_percent": float(100 * np.max(np.abs(width / width_exact - 1))),
        "max_width_error_percent_half_dx": float(100 * np.max(np.abs(width_fine / width_exact_fine - 1))),
        "energy_drift": rel_drift(ts["E_mean"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    viz.spacetime_heatmap(res, V=V.V, ax=axes[0], show_potential_panel=False,
                          title="|ψ(x,t)|²: packet moves at v_g and spreads")
    axes[1].plot(t / FS, width / NM, color=BLUE, label="simulation")
    axes[1].plot(t / FS, width_exact / NM, color=INK, linestyle="--", linewidth=1.2,
                 label="exact: σ(t) = σ₀√(1 + (ħt / 2mσ₀²)²)")
    axes[1].set(xlabel="t (fs)", ylabel="width σₓ (nm)",
                title=f"Spreading vs exact solution ({summary['free_particle']['max_width_error_percent']:.1f}% gap from grid spacing)")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[2].plot(t / FS, ts["T_mean"] / EV, color=BLUE, label="kinetic ⟨T⟩")
    axes[2].plot(t / FS, ts["E_mean"] / EV, color=INK, linestyle="--", linewidth=1.2, label="total ⟨H⟩")
    axes[2].set(xlabel="t (fs)", ylabel="energy (eV)", ylim=(0, 0.8),
                title="Energy is conserved (no potential to trade with)")
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("II.1  Free Gaussian wave packet (σ₀ = 2 nm, E₀ = 0.5 eV)", x=0.01, ha="left",
                  fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "part2_free_particle.png")
    plt.close(fig)
    if gifs:
        viz.make_gif(res, save_path=str(FIG / "01_free_gaussian.gif"), V=V.V, fps=20,
                     title="free Gaussian wave packet")


def superposition(summary: dict, gifs: bool) -> None:
    omega = 2 * np.pi * 3.7e13
    sigma_ho = np.sqrt(units.HBAR / (2 * units.M_E * omega))
    period = 2 * np.pi / omega
    # Same starting point as the notebook: the ground-state Gaussian displaced by 1.76 nm.
    # (The notebook labels this alpha = 2.5 because it set x = sqrt(2)*2.5*sigma; in the
    # standard convention <x> = 2*sigma*alpha, so this state has alpha = 1.77.)
    x_cl = np.sqrt(2) * 2.5 * sigma_ho
    alpha = x_cl / (2 * sigma_ho)
    # The packet is sigma = 0.5 nm wide and swings out to 1.76 nm, so its tail reaches past
    # 3 nm. The original run used walls at +/-3 nm, which clipped it (the source of the
    # ripples noted in the notebook); walls at +/-5 nm leave it untouched.
    g = gridmod.make_grid_nm(-5, 5, 801)
    V = potentials.harmonic(g, omega)
    dt = gridmod.visscher_dt_max(g, V_max=V.max_abs(g.x), safety=0.5)
    psi0 = gridmod.gaussian_packet(g, x0=x_cl, sigma=sigma_ho, k0=0.0)
    res = solvers.LeapfrogSolver(g, V, dt=dt, bc="hard").run(
        psi0, n_steps=int(3 * period / dt), n_save=240)
    ts = observables.timeseries(res, V_static=V.V)
    t = ts["t"]
    x_classical = x_cl * np.cos(omega * t)
    width = np.sqrt(ts["x_var"])

    # How many eigenstates make up this state? (it is a superposition of all of them)
    n = np.arange(0, 20)
    weights = np.exp(-alpha ** 2) * alpha ** (2 * n) / np.array([float(np.prod(range(1, k + 1))) for k in n])
    summary["superposition"] = {
        "alpha": round(alpha, 3),
        "mean_occupation_number": round(alpha ** 2, 3),
        "energy_expected_eV": units.HBAR * omega * (alpha ** 2 + 0.5) / EV,
        "energy_measured_eV": float(ts["E_mean"].mean() / EV),
        "largest_eigenstate_weights": {int(k): round(float(w), 3) for k, w in zip(n, weights) if w > 0.05},
        "max_deviation_from_classical_nm": float(np.max(np.abs(ts["x_mean"] - x_classical)) / NM),
        "classical_amplitude_nm": x_cl / NM,
        "width_change_percent": float(100 * (width.max() - width.min()) / width.mean()),
        "energy_drift": rel_drift(ts["E_mean"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    viz.spacetime_heatmap(res, V=V.V, ax=axes[0], show_potential_panel=False,
                          title="|ψ(x,t)|²: the packet rocks without spreading")
    axes[1].plot(t / FS, ts["x_mean"] / NM, color=BLUE, label="simulation ⟨x⟩")
    axes[1].plot(t / FS, x_classical / NM, color=INK, linestyle="--", linewidth=1.2,
                 label="classical x_cl cos ωt")
    axes[1].set(xlabel="t (fs)", ylabel="position (nm)", ylim=(-2.8, 2.4),
                title="The centre follows the classical motion")
    axes[1].legend(fontsize=8, loc="lower center", ncol=2)
    axes[2].plot(t / FS, ts["T_mean"] / EV, color=BLUE, label="kinetic ⟨T⟩")
    axes[2].plot(t / FS, ts["V_mean"] / EV, color=ORANGE, label="potential ⟨V⟩")
    axes[2].plot(t / FS, ts["E_mean"] / EV, color=INK, linestyle="--", linewidth=1.2, label="total ⟨H⟩")
    axes[2].set(xlabel="t (fs)", ylabel="energy (eV)", ylim=(0, 0.72),
                title="Kinetic and potential energy trade; total stays fixed")
    axes[2].legend(fontsize=8, loc="upper center", ncol=3)
    fig.suptitle(f"II.3  Superposition: harmonic-oscillator coherent state (α = {alpha:.2f}, ħω = "
                 f"{units.HBAR * omega / EV:.3f} eV)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "part2_superposition.png")
    plt.close(fig)
    if gifs:
        viz.make_gif(res, save_path=str(FIG / "01_coherent_state.gif"), V=V.V, fps=25,
                     title=f"coherent state |α={alpha:.2f}⟩")


# ----- Part III: tunnelling -----------------------------------------------------------------------

def barrier_setup(n_points: int, V0: float, a: float):
    g = gridmod.make_grid_nm(-50, 50, n_points)
    V = potentials.rectangular_barrier(g, -a / 2, a / 2, V0)
    bm = boundaries.absorbing_mask(g, sponge_width=6.0 * NM)
    return g, V, bm, V0 + np.max(np.abs(bm.V_imag))


def run_barrier(setup, E: float, sigma: float, n_save: int, t_end: float = 150 * FS):
    g, V, bm, vmax = setup
    dt = gridmod.visscher_dt_max(g, V_max=vmax, safety=0.5)
    psi0 = gridmod.gaussian_packet(g, x0=-15 * NM, sigma=sigma, k0=units.wavenumber_from_energy(E))
    return solvers.LeapfrogSolver(g, V, dt=dt, bc="absorbing", boundary_mask=bm).run(
        psi0, n_steps=int(t_end / dt), n_save=n_save, track_energy=False)


def tunnelling(summary: dict, gifs: bool) -> None:
    V0, a = 2.0 * EV, 0.6 * NM
    coarse = barrier_setup(2001, V0, a)
    fine = barrier_setup(4001, V0, a)
    g, V = coarse[0], coarse[1]

    demo = run_barrier(coarse, 1.0 * EV, 2.0 * NM, 200, t_end=120 * FS)
    if gifs:
        viz.make_gif(demo, save_path=str(FIG / "02_tunnelling.gif"), V=V.V, fps=20,
                     title="tunnelling, E₀ = 1 eV, V₀ = 2 eV")

    # Exact T(E) from the transfer matrix, for the same barrier
    g_tm = gridmod.make_grid_nm(-5, 5, 2001)
    V_tm = potentials.rectangular_barrier(g_tm, -a / 2, a / 2, V0)
    E_curve, T_curve, _ = analytical.transfer_matrix_T_curve(g_tm.x, V_tm.V, 0.05 * EV, 6.0 * EV,
                                                            n_points=400)
    T_formula = np.array([analytical.barrier_transmission(E, V0, a) for E in E_curve])

    sigma = 1.5 * NM

    def exact_for_packet(E: float) -> float:
        """A packet carries a spread of energies, so compare with T(E) averaged over it."""
        k0 = units.wavenumber_from_energy(E)
        k = np.linspace(k0 - 2 / sigma, k0 + 2 / sigma, 161)
        k = k[k > 0]
        w = np.exp(-2 * ((k - k0) * sigma) ** 2)          # |phi(k)|^2 of the Gaussian
        Ek = (units.HBAR * k) ** 2 / (2 * units.M_E)
        return float(np.sum(w * np.interp(Ek, E_curve, T_curve)) / np.sum(w))

    energies = np.array([0.5, 0.9, 1.2, 1.6, 1.9, 2.1, 2.5, 3.0, 4.0, 5.0]) * EV
    rows = []
    for E in energies:
        T_packet = observables.transmitted_probability(run_barrier(fine, E, sigma, 1500), 5 * NM)
        T_ref = exact_for_packet(E)
        rows.append({"E_eV": round(float(E / EV), 2), "T_simulation": T_packet,
                     "T_exact_for_packet": T_ref, "ratio": T_packet / T_ref})
    # Grid convergence: halving dx should shrink the error about 4x (second-order scheme)
    convergence = []
    for E in (1.6 * EV, 2.1 * EV):
        T_ref = exact_for_packet(E)
        T_c = observables.transmitted_probability(run_barrier(coarse, E, sigma, 1500), 5 * NM)
        T_f = next(r["T_simulation"] for r in rows if abs(r["E_eV"] - E / EV) < 1e-9)
        convergence.append({"E_eV": round(float(E / EV), 2),
                            "error_percent_dx_0.05nm": 100 * (T_c / T_ref - 1),
                            "error_percent_dx_0.025nm": 100 * (T_f / T_ref - 1)})
    summary["tunnelling"] = {"V0_eV": 2.0, "width_nm": 0.6, "packet_sigma_nm": 1.5,
                             "dx_nm": fine[0].dx / NM, "points": rows, "grid_convergence": convergence}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
    viz.spacetime_heatmap(demo, V=V.V, ax=axes[0], show_potential_panel=False,
                          title="|ψ(x,t)|²: E₀ = 1 eV hits a 2 eV barrier; a faint part gets through")
    ax = axes[1]
    ax.semilogy(E_curve / EV, np.maximum(T_curve, 1e-12), color=BLUE, label="transfer matrix (exact)")
    ax.semilogy(E_curve / EV, np.maximum(T_formula, 1e-12), color=INK, linestyle="--", linewidth=1.1,
                label="textbook formula (Griffiths)")
    ax.semilogy([r["E_eV"] for r in rows], [r["T_simulation"] for r in rows], "o", color=ORANGE,
                markersize=7, markeredgecolor="white", label="wave-packet simulation")
    ax.axvline(2.0, color=INK2, linestyle=":", linewidth=1)
    ax.text(2.05, 2e-6, "V₀ = 2 eV", color=INK2, fontsize=9)
    ax.set(xlabel="incident energy E₀ (eV)", ylabel="transmission T  (log)", ylim=(1e-6, 2),
           title="Transmission vs energy: simulation agrees with exact theory")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("III.1  Quantum tunnelling through a rectangular barrier (V₀ = 2 eV, a = 0.6 nm)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "part3_tunnelling.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gifs", action="store_true", help="also redraw the animated GIFs (slower)")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    summary: dict = {}
    part1(summary)
    free_particle(summary, args.gifs)
    superposition(summary, args.gifs)
    tunnelling(summary, args.gifs)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
