"""Plotting helpers: animations, space-time heatmaps, and conservation panels.

All routines accept the SI-unit :class:`SolverResult` plus a
:class:`tdse.units.DisplayUnits` instance, and emit figures with axes
labelled in the chosen display units (default nm / fs / eV).

Convention for space-time heatmaps
-----------------------------------
We follow the *spacetime-diagram* convention familiar from special
relativity: **time on the horizontal axis, position on the vertical axis**.
A free particle's trajectory then reads top-left to bottom-right (or
left-to-right at constant slope) just like a worldline.  All ``imshow``
calls in this module use ``Z`` of shape ``(N_x, N_t)`` so that rows are
position bins and columns are timesteps.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from . import units, observables
from .grid import Grid1D


# ============================================================================
# Static snapshot of psi (Re, Im, |psi|^2) with potential overlay
# ============================================================================

def plot_density_snapshot(
    psi: np.ndarray,
    grid: Grid1D,
    V: Optional[np.ndarray] = None,
    disp: units.DisplayUnits = units.DEFAULT_UNITS,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.0, 4.2),
    show_components: bool = True,
):
    """Single-snapshot plot of |psi|^2 (filled) plus Re psi, Im psi, and V(x).

    The probability density is filled in light green; Re psi and Im psi are
    drawn as thin lines on the *left* axis; V(x) is drawn on a *right* axis
    in eV with a shaded fill so it is unambiguous where the potential lives.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_disp = grid.x / disp.length
    # display rho in 1/length_unit so the y-axis reads natural numbers
    # (psi is normalised in SI so int |psi|^2 dx_SI = 1; rescaling x by
    #  disp.length gives int rho_disp d x_disp = 1).
    rho = np.abs(psi) ** 2 * disp.length
    rho_max = float(np.max(rho)) if rho.size else 1.0
    amp_max = max(float(np.max(np.abs(psi.real))),
                  float(np.max(np.abs(psi.imag)))) if psi.size else 1.0

    # Probability density (filled)
    ax.fill_between(x_disp, 0.0, rho, color="tab:green", alpha=0.35,
                    label=r"$|\psi(x)|^2$")
    ax.plot(x_disp, rho, color="tab:green", lw=2.0)

    if show_components:
        # Re psi, Im psi (rescaled onto the rho axis so they share the y-range)
        scale = rho_max / amp_max if amp_max > 0 else 1.0
        ax.plot(x_disp, psi.real * scale, color="tab:blue", lw=1.0,
                alpha=0.85, label=r"$\mathrm{Re}\,\psi$ (scaled)")
        ax.plot(x_disp, psi.imag * scale, color="tab:orange", lw=1.0,
                alpha=0.85, label=r"$\mathrm{Im}\,\psi$ (scaled)")

    ax.set_xlabel(f"x [{disp.length_label}]")
    ax.set_ylabel(r"$|\psi(x)|^2$ [1/" + disp.length_label + "]")
    ax.set_xlim(x_disp[0], x_disp[-1])
    ax.set_ylim(-0.05 * rho_max, 1.15 * rho_max if rho_max > 0 else 1.0)
    ax.grid(True, alpha=0.3)

    if V is not None:
        ax_v = ax.twinx()
        Vd = V / disp.energy
        ax_v.fill_between(x_disp, Vd, np.max(Vd) + 1.0,
                          color="0.65", alpha=0.20, lw=0)
        ax_v.plot(x_disp, Vd, color="black", lw=1.4, alpha=0.85)
        ax_v.set_ylabel(f"V(x) [{disp.energy_label}]", color="black")
        Vmin, Vmax = float(np.min(Vd)), float(np.max(Vd))
        pad = 0.1 * (Vmax - Vmin if Vmax > Vmin else 1.0)
        ax_v.set_ylim(Vmin - pad, Vmax + pad)
        ax_v.tick_params(axis="y", labelcolor="black")

    if title:
        ax.set_title(title)
    if show_components:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    return fig, ax


# ============================================================================
# Space-time heatmap.  *t-horizontal, x-vertical* (worldline convention)
# ============================================================================

def spacetime_heatmap(
    result,
    disp: units.DisplayUnits = units.DEFAULT_UNITS,
    quantity: str = "density",
    V: Optional[np.ndarray] = None,
    title: str = "",
    figsize: Tuple[float, float] = (9.5, 5.0),
    cmap_density: str = "viridis",
    ax: Optional[plt.Axes] = None,
    show_potential_panel: bool = True,
):
    """Heatmap with **time on horizontal axis, position on vertical axis**.

    Parameters
    ----------
    quantity : {'density', 'real', 'imag', 'phase'}
    show_potential_panel : if True and ``V`` is given, draw a thin V(x)
        panel to the *left* of the main heatmap so the geometry of the
        potential and the propagation are aligned vertically.
    """
    grid = result.grid
    x_disp = grid.x / disp.length
    t_disp = result.times / disp.time

    if quantity == "density":
        Z = (np.abs(result.psi) ** 2).T * disp.length   # shape (N_x, N_t)
        cmap = cmap_density
        cb_label = r"$|\psi(x,t)|^2$  [1/" + disp.length_label + "]"
        vmin = 0.0
        vmax = float(np.max(Z)) if Z.size else 1.0
    elif quantity == "real":
        Z = result.psi.real.T
        cmap = "RdBu_r"
        cb_label = r"$\mathrm{Re}\,\psi(x,t)$"
        vlim = float(np.max(np.abs(Z))) if Z.size else 1.0
        vmin, vmax = -vlim, vlim
    elif quantity == "imag":
        Z = result.psi.imag.T
        cmap = "RdBu_r"
        cb_label = r"$\mathrm{Im}\,\psi(x,t)$"
        vlim = float(np.max(np.abs(Z))) if Z.size else 1.0
        vmin, vmax = -vlim, vlim
    elif quantity == "phase":
        Z = np.angle(result.psi).T
        cmap = "twilight"
        cb_label = r"$\arg\,\psi(x,t)$"
        vmin, vmax = -np.pi, np.pi
    else:
        raise ValueError(f"unknown quantity {quantity!r}")

    have_panel = show_potential_panel and (V is not None) and ax is None
    if ax is None:
        if have_panel:
            fig, axes = plt.subplots(
                1, 2, figsize=figsize,
                gridspec_kw={"width_ratios": [1.0, 6.0], "wspace": 0.05},
                sharey=True,
            )
            ax_V, ax = axes
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax_V = None
    else:
        fig = ax.figure
        ax_V = None

    extent = [t_disp[0], t_disp[-1], x_disp[0], x_disp[-1]]
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cb_label)
    ax.set_xlabel(f"t [{disp.time_label}]")
    if ax_V is None:
        ax.set_ylabel(f"x [{disp.length_label}]")
    if title:
        ax.set_title(title)

    if ax_V is not None and V is not None:
        Vd = V / disp.energy
        ax_V.fill_betweenx(x_disp, np.min(Vd), Vd,
                           color="0.65", alpha=0.30, lw=0)
        ax_V.plot(Vd, x_disp, color="black", lw=1.4)
        ax_V.set_ylabel(f"x [{disp.length_label}]")
        ax_V.set_xlabel(f"V(x) [{disp.energy_label}]")
        ax_V.invert_xaxis()                # so V increases towards heatmap
        ax_V.grid(True, alpha=0.3)

    return fig, ax


# ============================================================================
# Animation + GIF export
# ============================================================================

def animate_wavefunction(
    result,
    V: Optional[object] = None,
    disp: units.DisplayUnits = units.DEFAULT_UNITS,
    interval_ms: int = 50,
    fps: int = 20,
    save_path: Optional[str] = None,
    title: str = "",
    show_components: bool = True,
    figsize: Tuple[float, float] = (7.0, 3.6),
    max_frames: Optional[int] = 80,
    dpi: int = 80,
    xlim: Optional[Tuple[float, float]] = None,
):
    """Animate Re psi, Im psi, |psi|^2, and (optionally) V over time.

    If ``save_path`` ends in ``.gif`` the animation is written out via
    Pillow; if it ends in ``.mp4`` we use ffmpeg.  If ``save_path`` is
    ``None`` the animation object is returned for inline notebook display.
    
    Parameters
    ----------
    xlim : tuple of float, optional
        (xmin, xmax) to set the x-axis limits. If None, uses full grid range.
    """
    grid = result.grid
    x_disp = grid.x / disp.length

    # Optionally subsample frames to keep GIF size reasonable
    psi_full = result.psi
    times_full = result.times
    n_full = len(psi_full)
    if max_frames is not None and n_full > max_frames:
        idx = np.linspace(0, n_full - 1, max_frames).astype(int)
        psi = psi_full[idx]
        times = times_full[idx]
    else:
        psi = psi_full
        times = times_full
    times_disp = times / disp.time

    rho = np.abs(psi) ** 2 * disp.length
    rho_max = float(np.max(rho)) if rho.size else 1.0
    amp_max = max(float(np.max(np.abs(psi.real))),
                  float(np.max(np.abs(psi.imag)))) if psi.size else 1.0
    if not np.isfinite(amp_max) or amp_max == 0.0:
        amp_max = 1.0
    if not np.isfinite(rho_max) or rho_max == 0.0:
        rho_max = 1.0
    scale = rho_max / amp_max

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(f"x [{disp.length_label}]")
    ax.set_ylabel(r"$|\psi|^2$  [1/" + disp.length_label + r"]")
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(x_disp[0], x_disp[-1])
    ax.set_ylim(-1.15 * rho_max, 1.25 * rho_max)
    ax.grid(True, alpha=0.3)

    V_is_time_dependent = bool(getattr(V, "is_time_dependent", False)) or (
        callable(V) and not hasattr(V, "V")
    )
    line_V = None
    if V is not None:
        ax_v = ax.twinx()
        V0 = V(grid.x, times[0]) if V_is_time_dependent else (
            V(grid.x, 0.0) if callable(V) else V
        )
        Vd = V0 / disp.energy
        if V_is_time_dependent:
            line_V, = ax_v.plot(x_disp, Vd, color="black", lw=1.2, alpha=0.9)
        else:
            ax_v.fill_between(x_disp, Vd, np.max(Vd) + 1.0,
                              color="0.65", alpha=0.20, lw=0)
            line_V, = ax_v.plot(x_disp, Vd, color="black", lw=1.2, alpha=0.85)
        ax_v.set_ylabel(f"V(x) [{disp.energy_label}]", color="black")
        if V_is_time_dependent:
            V_samples = np.asarray([V(grid.x, t) for t in times])
            Vmin, Vmax = float(np.min(V_samples / disp.energy)), float(np.max(V_samples / disp.energy))
        else:
            Vmin, Vmax = float(np.min(Vd)), float(np.max(Vd))
        pad = 0.1 * (Vmax - Vmin if Vmax > Vmin else 1.0)
        ax_v.set_ylim(Vmin - pad, Vmax + pad)

    fill = ax.fill_between(x_disp, 0.0, rho[0], color="tab:green",
                           alpha=0.35)
    line_rho, = ax.plot(x_disp, rho[0], color="tab:green", lw=2.0,
                        label=r"$|\psi|^2$")
    if show_components:
        line_re, = ax.plot(x_disp, psi[0].real * scale, color="tab:blue",
                           lw=1.0, alpha=0.85,
                           label=r"$\mathrm{Re}\,\psi$ (scaled)")
        line_im, = ax.plot(x_disp, psi[0].imag * scale, color="tab:orange",
                           lw=1.0, alpha=0.85,
                           label=r"$\mathrm{Im}\,\psi$ (scaled)")

    title_text = ax.set_title(
        f"{title}    t = {times_disp[0]:7.2f} {disp.time_label}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    def _frame(i):
        nonlocal fill
        # Re-build fill_between (matplotlib doesn't have set_data for it)
        fill.remove()
        fill = ax.fill_between(x_disp, 0.0, rho[i],
                               color="tab:green", alpha=0.35)
        line_rho.set_ydata(rho[i])
        if show_components:
            line_re.set_ydata(psi[i].real * scale)
            line_im.set_ydata(psi[i].imag * scale)
        if V is not None and V_is_time_dependent:
            Vd = V(grid.x, times[i]) / disp.energy
            line_V.set_ydata(Vd)
        title_text.set_text(
            f"{title}    t = {times_disp[i]:7.2f} {disp.time_label}")
        return ()

    anim = animation.FuncAnimation(fig, _frame, frames=len(psi),
                                   interval=interval_ms, blit=False)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".",
                    exist_ok=True)
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps, dpi=dpi,
                      savefig_kwargs={"facecolor": "white"})
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi)

    return fig, anim


def make_gif(
    result,
    save_path: str,
    V: Optional[np.ndarray] = None,
    disp: units.DisplayUnits = units.DEFAULT_UNITS,
    fps: int = 20,
    title: str = "",
    show_components: bool = True,
    figsize: Tuple[float, float] = (7.0, 3.6),
    max_frames: Optional[int] = 80,
    dpi: int = 80,
    close: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
):
    """Convenience wrapper around :func:`animate_wavefunction` that always
    writes a GIF and (by default) closes the matplotlib figure afterwards
    so it does not appear inline in the notebook on top of the saved file.

    Returns the absolute path of the GIF written.
    
    Parameters
    ----------
    xlim : tuple of float, optional
        (xmin, xmax) to set the x-axis limits. If None, uses full grid range.
    """
    fig, _ = animate_wavefunction(
        result, V=V, disp=disp, fps=fps, save_path=save_path,
        title=title, show_components=show_components, figsize=figsize,
        max_frames=max_frames, dpi=dpi, xlim=xlim,
    )
    if close:
        plt.close(fig)
    return os.path.abspath(save_path)


# ============================================================================
# Conservation diagnostics (norm, energy, momentum)
# ============================================================================

def plot_norm_conservation(
    result,
    disp: units.DisplayUnits = units.DEFAULT_UNITS,
    figsize: Tuple[float, float] = (10.0, 3.6),
    title: str = "Probability conservation",
):
    """Two-panel: (left) norms vs t; (right) log relative drift vs t.

    Always plots ``int |psi|^2 dx``.  If the run is a leapfrog we also
    overlay the *Visscher modified norm* (the actual conserved quantity).
    The right panel uses a log scale so the difference between
    ``machine epsilon`` (Visscher) and ``O(dt^2)`` (ordinary) is visible.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    t = result.diag_times / disp.time

    n0 = result.norm[0]
    # Thick semi-transparent ordinary-norm trace + thin dashed Visscher line
    # with markers, so the latter is visible even when both lines coincide
    # to within the y-axis padding.
    axes[0].plot(t, result.norm, color="tab:blue", lw=3.0, alpha=0.45,
                 label=r"$\int|\psi|^2\,dx$  (ordinary)")
    if result.norm_visscher is not None:
        n_mk = max(1, len(t) // 25)
        axes[0].plot(t, result.norm_visscher, color="tab:red", lw=1.4,
                     ls="--", marker="o", markersize=4, markevery=n_mk,
                     markerfacecolor="white", markeredgewidth=1.0,
                     label=r"$N_{\mathrm{Visscher}}$")
    axes[0].axhline(1.0, color="0.6", lw=0.8, alpha=0.5)
    axes[0].set_xlabel(f"t [{disp.time_label}]")
    axes[0].set_ylabel("norm")
    axes[0].set_title("Norm value vs time")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(min(0.999 * n0, axes[0].get_ylim()[0]),
                     max(1.001 * n0, axes[0].get_ylim()[1]))

    err = np.abs(result.norm - n0) / n0
    axes[1].semilogy(t, np.maximum(err, 1e-18), color="tab:blue",
                     lw=1.4, label=r"$|\!\int|\psi|^2 - 1|$")
    if result.norm_visscher is not None:
        v0 = result.norm_visscher[0]
        verr = np.abs(result.norm_visscher - v0) / v0
        axes[1].semilogy(t, np.maximum(verr, 1e-18), color="tab:red",
                         lw=1.4, ls="--",
                         label=r"$|N_{\mathrm{Visscher}}/N_0 - 1|$")
    axes[1].axhline(np.finfo(float).eps, color="0.6", lw=0.8, ls=":",
                    label=r"machine $\epsilon$")
    axes[1].set_xlabel(f"t [{disp.time_label}]")
    axes[1].set_ylabel("relative drift")
    axes[1].set_title("Log relative drift vs time")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3, which="both")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_energy_decomposition(
    result,
    V_static: Optional[np.ndarray] = None,
    disp: units.DisplayUnits = units.DEFAULT_UNITS,
    m: float = units.M_E,
    figsize: Tuple[float, float] = (10.0, 3.6),
    title: str = "Energy conservation",
):
    """Two-panel: (left) <T>, <V>, <H> vs t; (right) drift of <H> vs t.

    The kinetic and potential expectation values are computed from the
    *saved snapshots* (``result.psi``).  ``V_static`` is the static
    potential to use; for time-dependent runs pass ``V(x, t_save[k])``
    inside a small loop instead (or pass ``None`` and we will only plot
    the total <H> from ``result.energy``).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    t_save = result.times / disp.time
    t_diag = result.diag_times / disp.time
    grid = result.grid

    if V_static is not None:
        T_t = observables.kinetic_energy(result.psi, grid.dx, m=m)
        V_t = observables.potential_energy(result.psi, V_static, grid.dx)
        H_t = T_t + V_t
        n_mk = max(1, len(t_save) // 20)
        # H first as a thick semi-transparent band, then T (orange, marked)
        # and V (green, dashed) on top so all three are individually visible
        # even when V == 0 (free particle) makes T == H exactly.
        axes[0].plot(t_save, H_t / disp.energy, label=r"$\langle H\rangle$",
                     color="tab:blue", lw=3.0, alpha=0.45)
        axes[0].plot(t_save, T_t / disp.energy, label=r"$\langle T\rangle$",
                     color="tab:orange", lw=1.5,
                     marker="s", markersize=4, markevery=n_mk,
                     markerfacecolor="white", markeredgewidth=1.0)
        axes[0].plot(t_save, V_t / disp.energy, label=r"$\langle V\rangle$",
                     color="tab:green", lw=1.5, ls=(0, (4, 2)))
    elif result.energy is not None:
        axes[0].plot(t_diag, result.energy / disp.energy,
                     label=r"$\langle H\rangle$",
                     color="tab:blue", lw=1.4)

    axes[0].set_xlabel(f"t [{disp.time_label}]")
    axes[0].set_ylabel(f"energy [{disp.energy_label}]")
    axes[0].set_title("Kinetic + potential = total")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    if result.energy is not None:
        E0 = result.energy[0]
        if abs(E0) < 1e-30:
            E0 = 1.0
        drift = (result.energy - result.energy[0]) / disp.energy * 1e3   # meV
        axes[1].plot(t_diag, drift, color="tab:purple", lw=1.4,
                     label=r"$\langle H\rangle(t) - \langle H\rangle(0)$")
        axes[1].axhline(0.0, color="0.5", lw=0.8)
        axes[1].set_xlabel(f"t [{disp.time_label}]")
        axes[1].set_ylabel(r"$\Delta\langle H\rangle$ [meV]")
        axes[1].set_title("Energy drift (meV scale)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_observables_panel(
    result,
    V_static: Optional[np.ndarray] = None,
    disp: units.DisplayUnits = units.DEFAULT_UNITS,
    m: float = units.M_E,
    figsize: Tuple[float, float] = (12.5, 7.0),
    title: str = "Observable conservation diagnostics",
    classical_x=None,
    classical_p=None,
):
    """Big six-panel diagnostics plot:

    * (0,0)  norm conservation (linear)
    * (0,1)  log relative drift of all conserved norms
    * (0,2)  total energy (and KE, PE)
    * (1,0)  <x>(t)  with optional classical reference
    * (1,1)  <p>(t)  with optional classical reference
    * (1,2)  position uncertainty sigma_x(t)

    The bottom row uses *snapshot* times ``result.times``; the top row
    uses *diagnostic* times ``result.diag_times`` (every step).
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    grid = result.grid
    dx = grid.dx
    x = grid.x

    t_diag = result.diag_times / disp.time
    t_save = result.times / disp.time

    # --- (0,0) norm conservation, linear scale ----------------------------
    # We plot the *drift from t=0*,  N(t) - N(0),  rather than N(t) - 1,
    # so both curves start at exactly 0 and the eye reads the size of the
    # actual drift directly.
    #
    # Both norms are drawn:
    #   * blue thick band  : ordinary norm  N_ord = sum |psi|^2 dx
    #   * red dashed       : Visscher norm  N_v   = sum [R^2 + I_- I_+] dx
    # The Visscher norm has a known O(dt^2) constant offset from N_ord
    # coming from the half-step bootstrap
    #   I^{+/-1/2} = I^0 -/+ (dt/2hbar) H R^0
    # (analytically  N_v(0) - N_ord(0) = -(dt/2hbar)^2 <R0|H^2|R0>).
    # That offset is a *baseline* feature of the scheme, not a physics
    # error: it would make N_v(t)-1 sit pinned at the offset and visually
    # dominate any "N(t)-1" plot, falsely suggesting the conserved
    # quantity is the worse-behaved one.  By subtracting each baseline
    # we show what each scheme actually does about its own baseline:
    # N_ord drifts at O(dt^2) and N_v is conserved to machine epsilon.
    # If the simulation truly conserves probability the two drift curves
    # overlap on top of the y=0 line.
    n0 = result.norm[0]
    dev_ord = result.norm - n0
    dev_v = None
    drift = float(np.max(np.abs(dev_ord)))
    if result.norm_visscher is not None:
        v0 = result.norm_visscher[0]
        dev_v = result.norm_visscher - v0
        drift = max(drift, float(np.max(np.abs(dev_v))))
    pad = max(1e-15, 1.5 * drift)
    # Choose a "nice" decadal scale so y-axis ticks are O(1) numbers.
    decade = int(np.floor(np.log10(max(pad, 1e-15))))
    scale = 10.0 ** decade

    ax = axes[0, 0]
    ax.plot(t_diag, dev_ord / scale, color="tab:blue", lw=3.6, alpha=0.55,
            label=r"$N_{\mathrm{ord}}(t) - N_{\mathrm{ord}}(0)$")
    if dev_v is not None:
        n_mk = max(1, len(t_diag) // 25)
        ax.plot(t_diag, dev_v / scale,
                color="red", lw=1.4, ls="--",
                marker="o", markersize=4, markevery=n_mk,
                markerfacecolor="white", markeredgewidth=1.0, zorder=5,
                label=r"$N_{\mathrm{Visscher}}(t) - N_{\mathrm{Visscher}}(0)$")
        v_abs = float(np.median(result.norm_visscher))
        ax.text(
            0.98, 0.04,
            rf"baselines: $N_{{\mathrm{{ord}}}}(0)={n0:.6f}$,"
            rf"  $N_{{\mathrm{{Visscher}}}}(0)={v_abs:.6f}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="0.25",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="0.7", lw=0.6, alpha=0.85),
        )
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_ylim(-pad / scale, pad / scale)
    ax.set_xlabel(f"t [{disp.time_label}]")
    exp_str = (f"$10^{{{decade}}}$" if decade != 0 else "1")
    ax.set_ylabel(rf"$N(t) - N(0)$   [units of {exp_str}]")
    ax.set_title("Probability conservation  ($N$ should be constant)")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)

    # --- (0,1) relative drift, log scale ----------------------------------
    ax = axes[0, 1]
    err = np.abs(result.norm - n0) / n0
    ax.semilogy(t_diag, np.maximum(err, 1e-18), color="tab:blue",
                lw=1.3, label=r"ordinary norm")
    if result.norm_visscher is not None:
        v0 = result.norm_visscher[0]
        verr = np.abs(result.norm_visscher - v0) / v0
        ax.semilogy(t_diag, np.maximum(verr, 1e-18), color="tab:red",
                    lw=1.3, ls="--", label=r"Visscher $N$")
    ax.axhline(np.finfo(float).eps, color="0.6", lw=0.8, ls=":",
               label=r"machine $\epsilon$")
    ax.set_xlabel(f"t [{disp.time_label}]")
    ax.set_ylabel(r"$|\Delta N|/N_0$")
    ax.set_title("Log relative norm drift")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3, which="both")

    # --- (0,2) energy decomposition ---------------------------------------
    # Drawing order matters: <H> is plotted first as a thick semi-transparent
    # blue band so that <T> (orange) and <V> (green) remain visible *on top*
    # even in the free-particle limit V == 0 where T = H exactly.  Both <T>
    # and <V> use distinct dash patterns so that they read clearly when
    # they coincide with <H> (free particle: T == H; bound flat-bottom
    # well: V == 0 == H baseline).
    ax = axes[0, 2]
    if V_static is not None:
        Tt = observables.kinetic_energy(result.psi, dx, m=m)
        Vt = observables.potential_energy(result.psi, V_static, dx)
        Ht = Tt + Vt
        # Thick semi-transparent <H> band underneath; long-dash <T> on top
        # with substantial gaps so the blue band shows clearly between
        # dashes (this matters in the free-particle case where T == H);
        # short-dotted <V> for further visual contrast.
        ax.plot(t_save, Ht / disp.energy, color="tab:blue", lw=4.0,
                alpha=0.55,
                label=r"$\langle H\rangle = \langle T\rangle+\langle V\rangle$")
        ax.plot(t_save, Tt / disp.energy, color="tab:orange", lw=1.8,
                ls=(0, (10, 6)), label=r"$\langle T\rangle$")
        ax.plot(t_save, Vt / disp.energy, color="tab:green", lw=1.5,
                ls=(0, (1, 2)), label=r"$\langle V\rangle$")
    elif result.energy is not None:
        ax.plot(t_diag, result.energy / disp.energy,
                color="tab:blue", lw=1.3,
                label=r"$\langle H\rangle$")
    ax.set_xlabel(f"t [{disp.time_label}]")
    ax.set_ylabel(f"energy [{disp.energy_label}]")
    ax.set_title("Energy decomposition")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)

    # --- (1,0) <x>(t) with classical reference ----------------------------
    ax = axes[1, 0]
    xt = observables.position_expectation(result.psi, x, dx)
    ax.plot(t_save, xt / disp.length, color="tab:blue", lw=1.4,
            label=r"$\langle x\rangle$")
    if classical_x is not None:
        ax.plot(t_save, classical_x(result.times) / disp.length,
                color="tab:gray", lw=1.0, ls="--",
                label="classical")
    ax.set_xlabel(f"t [{disp.time_label}]")
    ax.set_ylabel(f"$\\langle x\\rangle$ [{disp.length_label}]")
    ax.set_title("Position expectation"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,1) <p>(t) with classical reference ----------------------------
    ax = axes[1, 1]
    pt = observables.momentum_expectation(result.psi, dx)
    p_unit = units.HBAR / disp.length          # hbar/nm  (a useful scale)
    ax.plot(t_save, pt / p_unit, color="tab:purple", lw=1.4,
            label=r"$\langle p\rangle$")
    if classical_p is not None:
        ax.plot(t_save, classical_p(result.times) / p_unit,
                color="tab:gray", lw=1.0, ls="--", label="classical")
    ax.set_xlabel(f"t [{disp.time_label}]")
    ax.set_ylabel(r"$\langle p\rangle$  [$\hbar$/" + disp.length_label + "]")
    ax.set_title("Momentum expectation"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,2) sigma_x(t) -------------------------------------------------
    ax = axes[1, 2]
    var = observables.position_variance(result.psi, x, dx)
    sig = np.sqrt(np.maximum(var, 0.0))
    ax.plot(t_save, sig / disp.length, color="tab:olive", lw=1.4,
            label=r"$\sigma_x(t)$")
    ax.set_xlabel(f"t [{disp.time_label}]")
    ax.set_ylabel(f"$\\sigma_x$ [{disp.length_label}]")
    ax.set_title("Position uncertainty"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, y=1.00)
    fig.tight_layout()
    return fig, axes


# Backwards-compatibility alias for code that still imports the old name
plot_diagnostics = plot_norm_conservation
