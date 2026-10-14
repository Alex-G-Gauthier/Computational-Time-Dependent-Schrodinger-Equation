"""Generate ODMR-style GIF animations for AC and DC Stark splitting.

Both GIFs use the same visual grammar pioneered by NV-diamond ODMR
spectroscopy: a 2-D heatmap of intensity vs (probe frequency, swept
parameter) on the left, and a live 1-D spectrum at the current
parameter value on the right with a numerical readout of the splitting.

Outputs
-------
* ``figures/03_ac_stark_splitting.gif`` -- drive amplitude rises 0 -> max,
                                            two dressed-state peaks pull
                                            apart linearly with $\\Omega_R$.
* ``figures/03_dc_stark_splitting.gif`` -- DC bias sweeps -max -> +max
                                            -> -max, two adiabatic
                                            eigenvalues come together at
                                            the avoided crossing then
                                            re-separate.

Run from the repo root::

    python tools/build_stark_gifs.py


===============================================================================
EDITING ROADMAP -- where to modify each of the 4 plot panels
===============================================================================

Each GIF is a 1x2 figure -> 2 panels per GIF -> 4 panels total.  Search this
file for the section banners below to jump to the code that draws each one.

  PLOT 1/4  AC Stark  -- LEFT  panel  (ODMR heatmap, fills row-by-row)
  PLOT 2/4  AC Stark  -- RIGHT panel  (current 1-D spectrum + splitting arrow)
  PLOT 3/4  DC Stark  -- LEFT  panel  (ODMR heatmap, sweep cursor)
  PLOT 4/4  DC Stark  -- RIGHT panel  (current 1-D spectrum + splitting arrow)

Quick-edit cheat sheet (most common knobs):

  Axis labels / titles / axis ranges
      -> AC_DEFAULTS dict (PLOT 1/4 + PLOT 2/4)        ~line 92
      -> DC_DEFAULTS dict (PLOT 3/4 + PLOT 4/4)        ~line 290
      OR pass keyword args to gif_ac_stark_splitting()
                              gif_dc_stark_splitting().

  Physics parameters (well width, drive amplitude, bias range, etc.)
      -> "PHYSICS PARAMETERS" block inside each function.

  Grid resolution, probe window, Lorentzian width
      -> "SPECTRUM GRID" block inside each function.

  Frame pacing / number of frames / GIF speed
      -> "FRAME PACING" block inside each function and the duration_ms
         argument to _save_gif() at the end of each function.

  Heatmap colormap, line colors, marker dots
      -> Inside the per-frame loop, search for the matching PLOT N/4 banner.

  Splitting-readout arrow + text (the red "<->" with the meV number)
      -> Inside the per-frame loop under each "RIGHT panel" PLOT N/4 banner.
===============================================================================
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from tdse import grid as gridmod, solvers, units  # noqa: E402

GIF_DIR = ROOT / "figures"
GIF_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["mathtext.fontset"] = "cm"
PLT_RC = {
    "figure.dpi": 90,
    "savefig.dpi": 90,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "semibold",
    "legend.fontsize": 9,
    "lines.linewidth": 1.4,
}


def _frame_to_pil(fig) -> Image.Image:
    """Render a matplotlib Figure to a PIL.Image via an in-memory PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def _save_gif(frames, out_path: Path, duration_ms: int) -> None:
    """Quantize each frame to an adaptive 256-colour palette then save GIF."""
    quantized = [f.convert("P", palette=Image.ADAPTIVE, dither=Image.NONE) for f in frames]
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


# ===========================================================================
# AC Stark / Autler-Townes splitting in a driven ISW (NB03 §3)
# Builds figures/03_ac_stark_splitting.gif (PLOT 1/4 + PLOT 2/4).
# ===========================================================================

# ---- AC defaults: text knobs for PLOT 1/4 and PLOT 2/4 ----------------------
# This dict is the FASTEST way to retitle / relabel either panel in the AC
# Stark GIF.  Edit a value here OR override at call time, e.g.::
#
#     gif_ac_stark_splitting(spectrum_title="my new title", spectrum_ylim=(0, 3))
#
# Keys map to panels as follows:
#   suptitle            -> figure-level title (spans both panels)
#   heatmap_*           -> PLOT 1/4 (left, ODMR heatmap)
#   spectrum_*          -> PLOT 2/4 (right, current 1-D spectrum)
#   spectrum_x/y lim    -> right-panel axis ranges (None -> auto from probe grid)
#
# spectrum_title is a Python format string; the placeholder ``{drive_meV}``
# is replaced per frame with the current value of hbar*Omega_R in meV.
AC_DEFAULTS = dict(
    suptitle="AC Stark / Autler-Townes splitting -- driven ISW",
    heatmap_xlabel=r"detuning from resonance  $\hbar(\omega_{\rm probe} - \omega_{12})$  [meV]",
    heatmap_ylabel=r"drive strength  $\hbar\Omega_R$  [meV]",
    heatmap_title="ODMR-style intensity map (fills in as drive grows)",
    spectrum_xlabel=r"Detuning Energy [meV]",
    spectrum_ylabel="Intensity [a.u.]",
    spectrum_title=r"Photoluminescence Spectrum $\hbar\Omega_R = {drive_meV:.1f}$ meV",
    spectrum_xlim=None,           # None -> auto from probe grid
    spectrum_ylim=(0.0, 1),
)


def gif_ac_stark_splitting(
    *,
    suptitle: str | None = None,
    heatmap_xlabel: str | None = None,
    heatmap_ylabel: str | None = None,
    heatmap_title: str | None = None,
    spectrum_xlabel: str | None = None,
    spectrum_ylabel: str | None = None,
    spectrum_title: str | None = None,
    spectrum_xlim: tuple[float, float] | None = None,
    spectrum_ylim: tuple[float, float] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Animate AC Stark splitting as the drive amplitude is ramped up.

    Frames sweep $\\Omega_R$ from 0 to its full project value (the F0 used
    in the resonant Rabi run of nb03 §3).  Each frame shows:

    * Left -- ODMR-style heatmap (intensity vs probe detuning vs drive
      strength) filled in row-by-row up to the current drive strength.
    * Right -- the current 1-D probe spectrum with a numerical readout of
      the AC Stark splitting, $\\hbar\\Omega_R$, in meV.

    All axis labels and titles can be overridden via keyword arguments.
    ``spectrum_title`` is a Python format string that may reference
    ``{drive_meV}`` (the current value of $\\hbar\\Omega_R$ in meV).
    Defaults live in :data:`AC_DEFAULTS` for easy editing.
    """
    cfg = dict(AC_DEFAULTS)
    overrides = dict(
        suptitle=suptitle,
        heatmap_xlabel=heatmap_xlabel,
        heatmap_ylabel=heatmap_ylabel,
        heatmap_title=heatmap_title,
        spectrum_xlabel=spectrum_xlabel,
        spectrum_ylabel=spectrum_ylabel,
        spectrum_title=spectrum_title,
        spectrum_xlim=spectrum_xlim,
        spectrum_ylim=spectrum_ylim,
    )
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    out = out_path if out_path is not None else GIF_DIR / "03_ac_stark_splitting.gif"

    # ---- PHYSICS PARAMETERS (AC Stark) -----------------------------------
    # Edit these to change well width, dipole drive, max Rabi frequency, etc.
    # The splitting_max_meV computed here drives both the heatmap y-axis
    # range (PLOT 1/4) and the maximum value the readout text can quote
    # (PLOT 2/4).  Keep these in sync with NB03 §3 to match the static plot.
    L = 2.0 * units.NM                              # ISW half-width L
    F0 = 0.15 * units.EV / units.NM                 # peak drive amplitude
    x12 = -16.0 * L / (9.0 * np.pi ** 2)            # |1>-|2> dipole matrix elt.
    Omega_R_max = F0 * abs(x12) / units.HBAR        # rad/s
    splitting_max_meV = units.HBAR * Omega_R_max / units.EV * 1000.0  # ~ 54 meV

    # ---- SPECTRUM GRID (AC Stark) ----------------------------------------
    # n_drive: number of rows in the heatmap (PLOT 1/4 vertical resolution).
    # n_probe: number of columns / right-panel x-resolution (PLOT 2/4).
    # probe_max_meV / gamma_meV: probe window and Lorentzian half-width.
    # Increasing n_drive or n_probe makes the GIF crisper but bigger on disk.
    n_drive = 80
    drive_meV = np.linspace(0.0, splitting_max_meV, n_drive)
    n_probe = 320
    probe_max_meV = 0.85 * splitting_max_meV
    probe_meV = np.linspace(-probe_max_meV, +probe_max_meV, n_probe)
    gamma_meV = 0.04 * splitting_max_meV            # Lorentzian half-width

    # Pre-compute the full intensity map I(probe, drive); each frame just
    # masks rows above the current drive (PLOT 1/4) and slices the current
    # row for the live spectrum (PLOT 2/4).
    #
    # The map is normalised so I_full.max() == 1.0 -- this makes the
    # spectrum panel fit cleanly inside spectrum_ylim=(0, 1) and keeps
    # separated peaks at ~0.5, leaving headroom for the splitting arrow.
    # The heatmap (PLOT 1/4) uses a relative colour scale so it looks the
    # same regardless of this overall scaling.
    P, OR = np.meshgrid(probe_meV, drive_meV)
    I_full = (
        gamma_meV ** 2 / ((P - OR / 2.0) ** 2 + gamma_meV ** 2)
        + gamma_meV ** 2 / ((P + OR / 2.0) ** 2 + gamma_meV ** 2)
    )
    I_full /= I_full.max()
    I_max = float(I_full.max())

    # ---- FRAME PACING (AC Stark) -----------------------------------------
    # n_ramp = frames during which the drive is rising 0 -> max.
    # n_hold = frames at the end that hold the saturated map.
    # GIF playback speed is set by ``duration_ms`` in _save_gif() at the
    # bottom of this function.
    n_ramp = 50
    n_hold = 12
    n_frames = n_ramp + n_hold

    # ---- PER-FRAME RENDER LOOP (AC Stark) --------------------------------
    frames = []
    for k in range(n_frames):
        if k < n_ramp:
            cur_idx = int(k * (n_drive - 1) / (n_ramp - 1))
        else:
            cur_idx = n_drive - 1
        cur_drive = drive_meV[cur_idx]              # current hbar*Omega_R [meV]

        I_masked = I_full.copy()
        I_masked[cur_idx + 1:, :] = np.nan          # rows above cursor are blank
        I_cur = I_full[cur_idx, :]                  # 1-D slice for the spectrum

        with plt.rc_context(PLT_RC):
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

            # =============================================================
            # PLOT 1/4  --  AC Stark, LEFT panel (ODMR heatmap)
            # Knobs: cmap (colormap), vmax (contrast), cyan cursor + dots,
            #        labels/title come from cfg["heatmap_*"].
            # =============================================================
            ax = axes[0]
            cmap = plt.get_cmap("magma").copy()     # heatmap colormap
            cmap.set_bad(color="#f0f0f2")           # colour for "not yet swept"
            ax.imshow(
                I_masked,
                extent=[probe_meV.min(), probe_meV.max(),
                        drive_meV.min(), drive_meV.max()],
                aspect="auto", origin="lower",
                cmap=cmap, vmin=0.0, vmax=I_max * 0.7,   # contrast cap
                interpolation="nearest",
            )
            ax.axhline(cur_drive, color="cyan", lw=1.2, alpha=0.85)  # cursor row
            ax.plot([-cur_drive / 2.0], [cur_drive], "o",
                    color="cyan", ms=7, mec="white", mew=1.0)        # left dot
            ax.plot([+cur_drive / 2.0], [cur_drive], "o",
                    color="cyan", ms=7, mec="white", mew=1.0)        # right dot
            ax.set_xlabel(cfg["heatmap_xlabel"])
            ax.set_ylabel(cfg["heatmap_ylabel"])
            ax.set_title(cfg["heatmap_title"])
            ax.tick_params(axis="both", which="both", direction="out")

            # =============================================================
            # PLOT 2/4  --  AC Stark, RIGHT panel (current 1-D spectrum)
            # Knobs: line/fill colour (tab:purple), zero-detuning guideline,
            #        red double-headed arrow + readout text below,
            #        axis ranges/labels/title come from cfg["spectrum_*"].
            # =============================================================
            ax2 = axes[1]
            ax2.fill_between(                                       # shaded under spectrum
                probe_meV, 0.0, I_cur, color="tab:purple", alpha=0.30
            )
            ax2.plot(probe_meV, I_cur, color="tab:purple", lw=2.0)  # spectrum line
            ax2.axvline(0.0, color="0.7", lw=0.6, ls=":")           # bare-line marker

            # ---- splitting readout (red <-> arrow + meV text) -----------
            # The arrow + text are positioned proportionally to the
            # right-panel ylim, so they always sit just below the top of
            # the plot regardless of the spectrum_ylim you pick:
            #   y_arrow      = 80%   of ylim_top
            #   text baseline= ~85%  of ylim_top  (y_arrow + 4.7% offset)
            # Tweak the 0.80 / 0.047 factors here to slide them up/down.
            ylim_top = cfg["spectrum_ylim"][1]
            y_arrow = 0.80 * ylim_top
            y_text = y_arrow + 0.047 * ylim_top
            if cur_drive > 0.04 * splitting_max_meV:
                ax2.annotate(
                    "",
                    xy=(+cur_drive / 2.0, y_arrow),
                    xytext=(-cur_drive / 2.0, y_arrow),
                    arrowprops=dict(
                        arrowstyle="<->", color="tab:red", lw=1.6,
                        shrinkA=0, shrinkB=0,
                    ),
                )
                ax2.text(
                    0.0, y_text,
                    rf"$\hbar\Omega_R = {cur_drive:.1f}$ meV",
                    ha="center", va="bottom",
                    fontsize=11, color="tab:red", fontweight="bold",
                )
            else:
                ax2.text(
                    0.0, y_text,
                    "no drive  --  bare line at $\\omega_{12}$",
                    ha="center", va="bottom",
                    fontsize=10, color="0.4",
                )

            # ---- right-panel axis ranges / labels / title ---------------
            xlim = cfg["spectrum_xlim"] or (probe_meV.min(), probe_meV.max())
            ax2.set_xlim(*xlim)
            ax2.set_ylim(*cfg["spectrum_ylim"])
            ax2.set_xlabel(cfg["spectrum_xlabel"])
            ax2.set_ylabel(cfg["spectrum_ylabel"])
            ax2.set_title(cfg["spectrum_title"].format(drive_meV=cur_drive))
            ax2.grid(alpha=0.25)

            # ---- figure-level suptitle (spans both panels) --------------
            fig.suptitle(
                cfg["suptitle"],
                fontsize=12, fontweight="bold",
            )
            fig.tight_layout()

            frames.append(_frame_to_pil(fig))
            plt.close(fig)

    # GIF playback speed (ms per frame) -- raise to slow the animation down.
    _save_gif(frames, out, duration_ms=80)
    return out


# ===========================================================================
# DC Stark / avoided-crossing splitting in LZ wells (NB03 §4)
# Builds figures/03_dc_stark_splitting.gif (PLOT 3/4 + PLOT 4/4).
# ===========================================================================

# ---- DC defaults: text knobs for PLOT 3/4 and PLOT 4/4 ----------------------
# This dict is the FASTEST way to retitle / relabel either panel in the DC
# Stark GIF.  Edit a value here OR override at call time, e.g.::
#
#     gif_dc_stark_splitting(spectrum_title="my new title", spectrum_xlim=(-50, 50))
#
# Keys map to panels as follows:
#   suptitle            -> figure-level title (spans both panels)
#   heatmap_*           -> PLOT 3/4 (left, ODMR heatmap with sweep cursor)
#   spectrum_*          -> PLOT 4/4 (right, current 1-D spectrum)
#   spectrum_x/y lim    -> right-panel axis ranges (None -> auto from probe grid)
#
# ``suptitle`` and ``spectrum_title`` are Python format strings; the
# placeholders below are replaced per frame with the current values:
#   {eps_meV}        -- current DC bias in meV
#   {E0_meV}         -- lower adiabatic eigenvalue at this bias [meV]
#   {E1_meV}         -- upper adiabatic eigenvalue at this bias [meV]
#   {split_meV}      -- E1 - E0 at this bias [meV]
#   {two_delta_meV}  -- 2*Delta = avoided-crossing gap at eps=0 [meV]
DC_DEFAULTS = dict(
    suptitle=(
        r"DC Stark / avoided-crossing splitting in LZ wells  "
        r"($2\Delta = {two_delta_meV:.1f}$ meV at $\varepsilon = 0$)"
    ),
    heatmap_xlabel=r"DC bias  $\varepsilon = \alpha t$  [meV]",
    heatmap_ylabel=r"probe energy  $\hbar\omega_{\rm probe}$  [meV]",
    heatmap_title="ODMR-style intensity map (sweep cursor)",
    spectrum_xlabel=r"Detuning Energy [meV]",
    spectrum_ylabel="Intensity [a.u.]",
    spectrum_title=r"Photoluminescence Spectrum $\varepsilon = {eps_meV:+.0f}$ meV",
    spectrum_xlim=None,           # None -> auto from probe grid
    spectrum_ylim=(0.0, 1),
)


def gif_dc_stark_splitting(
    *,
    suptitle: str | None = None,
    heatmap_xlabel: str | None = None,
    heatmap_ylabel: str | None = None,
    heatmap_title: str | None = None,
    spectrum_xlabel: str | None = None,
    spectrum_ylabel: str | None = None,
    spectrum_title: str | None = None,
    spectrum_xlim: tuple[float, float] | None = None,
    spectrum_ylim: tuple[float, float] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Animate DC Stark splitting as the bias sweeps through the avoided crossing.

    Frames sweep $\\varepsilon$ from $-\\varepsilon_{\\max}$ to
    $+\\varepsilon_{\\max}$ and back (one boomerang cycle).  Each frame shows:

    * Left -- ODMR-style heatmap of intensity vs (probe energy, DC bias),
      with a vertical sweep cursor at the current $\\varepsilon$ and dots
      marking the two adiabatic eigenvalues at that bias.
    * Right -- the current 1-D probe spectrum with the splitting
      $E_1(\\varepsilon) - E_0(\\varepsilon)$ annotated.

    All axis labels and titles can be overridden via keyword arguments.
    ``spectrum_title`` is a Python format string that may reference
    ``{eps_meV}``, ``{E0_meV}``, ``{E1_meV}``, ``{split_meV}`` (the current
    bias, the two adiabatic eigenvalues, and their gap, all in meV).
    ``suptitle`` may additionally use ``{two_delta_meV}``.
    Defaults live in :data:`DC_DEFAULTS` for easy editing.
    """
    cfg = dict(DC_DEFAULTS)
    overrides = dict(
        suptitle=suptitle,
        heatmap_xlabel=heatmap_xlabel,
        heatmap_ylabel=heatmap_ylabel,
        heatmap_title=heatmap_title,
        spectrum_xlabel=spectrum_xlabel,
        spectrum_ylabel=spectrum_ylabel,
        spectrum_title=spectrum_title,
        spectrum_xlim=spectrum_xlim,
        spectrum_ylim=spectrum_ylim,
    )
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    out = out_path if out_path is not None else GIF_DIR / "03_dc_stark_splitting.gif"

    # ---- PHYSICS PARAMETERS (DC Stark) -----------------------------------
    # Two Gaussian wells centred at xL/xR define the LZ double-well.  d0
    # sets the well depth and sigma the well width.  ``g`` is the spatial
    # grid we diagonalise on for each bias value.  Edit these to change
    # the position, depth, or smoothness of the avoided crossing.
    xL = -1.0 * units.NM                            # left  well centre
    xR = +1.0 * units.NM                            # right well centre
    sigma = 0.50 * units.NM                         # well width
    d0 = 0.50 * units.EV                            # well depth
    g = gridmod.make_grid_nm(-8, 8, 201)            # spatial grid
    GL = np.exp(-((g.x - xL) ** 2) / (2.0 * sigma ** 2))
    GR = np.exp(-((g.x - xR) ** 2) / (2.0 * sigma ** 2))

    # ---- AVOIDED-CROSSING GAP --------------------------------------------
    # Diagonalise V0 = -d0*(GL+GR) once to get 2*Delta -- the splitting at
    # zero bias.  This is the single number that anchors the y-axis range
    # of PLOT 3/4 and the splitting readout in PLOT 4/4.
    Es_static, _ = solvers.bound_eigenstates(
        g, -d0 * (GL + GR), n_states=2,
    )
    Delta = 0.5 * (Es_static[1] - Es_static[0])

    # ---- BIAS SWEEP (PLOT 3/4 horizontal axis) ---------------------------
    # eps_max sets the DC-bias range in units of Delta.  n_eps sets the
    # horizontal resolution of the heatmap (PLOT 3/4) and the number of
    # eigenvalue solves done up front.
    eps_max = 30.0 * Delta
    n_eps = 120
    eps_grid = np.linspace(-eps_max, +eps_max, n_eps)
    E0_eps = np.empty(n_eps)
    E1_eps = np.empty(n_eps)
    for k, eps in enumerate(eps_grid):
        V_eps = -d0 * (GL + GR) + 0.5 * eps * (GL - GR)
        Es, _ = solvers.bound_eigenstates(g, V_eps, n_states=2)
        E0_eps[k] = Es[0]
        E1_eps[k] = Es[1]

    meV = units.EV / 1000.0
    eps_meV = eps_grid / meV
    E0_meV = E0_eps / meV
    E1_meV = E1_eps / meV
    Delta_meV = Delta / meV

    # ---- SPECTRUM GRID (DC Stark) ----------------------------------------
    # n_probe / pad / gamma define the probe-energy axis (vertical axis of
    # PLOT 3/4, x-axis of PLOT 4/4) and the Lorentzian half-width.
    pad = 8.0
    probe_min = float(E0_meV.min()) - pad
    probe_max = float(E1_meV.max()) + pad
    n_probe = 380
    probe_meV = np.linspace(probe_min, probe_max, n_probe)
    gamma = 1.5                                     # Lorentzian half-width [meV]

    # Pre-compute the full intensity map I(bias, probe).
    # Normalised so I_full.max() == 1.0 (see AC version for rationale).
    # Keeps the right-panel spectrum inside spectrum_ylim=(0, 1) without
    # clipping when the two Lorentzians overlap at the avoided crossing.
    I_full = np.zeros((n_eps, n_probe))
    for k in range(n_eps):
        I_full[k, :] = (
            gamma ** 2 / ((probe_meV - E0_meV[k]) ** 2 + gamma ** 2)
            + gamma ** 2 / ((probe_meV - E1_meV[k]) ** 2 + gamma ** 2)
        )
    I_full /= I_full.max()
    I_max = float(I_full.max())

    # ---- FRAME PACING (DC Stark) -----------------------------------------
    # The cursor sweeps -eps_max -> +eps_max -> -eps_max ("boomerang").
    # n_half is the number of frames per leg.  Total frames = 2*n_half - 1.
    # GIF playback speed is set by ``duration_ms`` in _save_gif() at the
    # bottom of this function.
    n_half = 35
    fwd = np.linspace(0, n_eps - 1, n_half).astype(int)
    bwd = np.linspace(n_eps - 1, 0, n_half).astype(int)[1:]
    eps_traj_idx = np.concatenate([fwd, bwd])

    # ---- PER-FRAME RENDER LOOP (DC Stark) --------------------------------
    frames = []
    for cur_idx in eps_traj_idx:
        cur_eps = float(eps_meV[cur_idx])           # current bias [meV]
        cur_E0 = float(E0_meV[cur_idx])             # lower adiabatic E [meV]
        cur_E1 = float(E1_meV[cur_idx])             # upper adiabatic E [meV]
        cur_split = cur_E1 - cur_E0                 # current gap [meV]
        I_cur = I_full[cur_idx, :]                  # 1-D slice for spectrum

        with plt.rc_context(PLT_RC):
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

            # =============================================================
            # PLOT 3/4  --  DC Stark, LEFT panel (ODMR heatmap)
            # Knobs: cmap (colormap), vmax (contrast), cyan sweep cursor +
            #        eigenvalue dots, labels/title come from cfg["heatmap_*"].
            # =============================================================
            ax = axes[0]
            cmap = plt.get_cmap("magma")            # heatmap colormap
            ax.imshow(
                I_full.T,
                extent=[eps_meV.min(), eps_meV.max(),
                        probe_meV.min(), probe_meV.max()],
                aspect="auto", origin="lower",
                cmap=cmap, vmin=0.0, vmax=I_max * 0.7,   # contrast cap
                interpolation="nearest",
            )
            ax.axvline(cur_eps, color="cyan", lw=1.2, alpha=0.85)  # cursor col
            ax.plot([cur_eps], [cur_E0], "o",
                    color="cyan", ms=7, mec="white", mew=1.0)      # lower dot
            ax.plot([cur_eps], [cur_E1], "o",
                    color="cyan", ms=7, mec="white", mew=1.0)      # upper dot
            ax.set_xlabel(cfg["heatmap_xlabel"])
            ax.set_ylabel(cfg["heatmap_ylabel"])
            ax.set_title(cfg["heatmap_title"])
            ax.tick_params(axis="both", which="both", direction="out")

            # =============================================================
            # PLOT 4/4  --  DC Stark, RIGHT panel (current 1-D spectrum)
            # Knobs: line/fill colour (tab:purple), grey dashed eigenvalue
            #        guides, red double-headed arrow + readout text below,
            #        axis ranges/labels/title come from cfg["spectrum_*"].
            # =============================================================
            ax2 = axes[1]
            ax2.fill_between(                                       # shaded under spectrum
                probe_meV, 0.0, I_cur, color="tab:purple", alpha=0.30,
            )
            ax2.plot(probe_meV, I_cur, color="tab:purple", lw=2.0)  # spectrum line
            ax2.axvline(cur_E0, color="0.45", ls="--", lw=0.8)      # E0 guide
            ax2.axvline(cur_E1, color="0.45", ls="--", lw=0.8)      # E1 guide

            # ---- splitting readout (red <-> arrow + meV text) -----------
            # Arrow + text positions auto-scale with the right-panel ylim
            # so they always fit:
            #   y_arrow      = 80%  of ylim_top
            #   text baseline= ~85% of ylim_top  (y_arrow + 4.7% offset)
            # Tweak the 0.80 / 0.047 factors here to slide them up/down.
            ylim_top = cfg["spectrum_ylim"][1]
            y_arrow = 0.80 * ylim_top
            y_text = y_arrow + 0.047 * ylim_top
            ax2.annotate(
                "",
                xy=(cur_E1, y_arrow), xytext=(cur_E0, y_arrow),
                arrowprops=dict(
                    arrowstyle="<->", color="tab:red", lw=1.6,
                    shrinkA=0, shrinkB=0,
                ),
            )
            mid = 0.5 * (cur_E0 + cur_E1)
            ax2.text(
                mid, y_text,
                rf"$E_1 - E_0 = {cur_split:.1f}$ meV",
                ha="center", va="bottom",
                fontsize=11, color="tab:red", fontweight="bold",
            )

            # ---- right-panel axis ranges / labels / title ---------------
            # ``fmt`` exposes per-frame numbers to the title format strings.
            # Add new keys here if you want to use them in cfg["spectrum_title"]
            # or cfg["suptitle"] (e.g. {E0_meV}, {E1_meV}, {split_meV}, ...).
            fmt = dict(
                eps_meV=cur_eps,
                E0_meV=cur_E0,
                E1_meV=cur_E1,
                split_meV=cur_split,
                two_delta_meV=2.0 * Delta_meV,
            )
            xlim = cfg["spectrum_xlim"] or (probe_meV.min(), probe_meV.max())
            ax2.set_xlim(*xlim)
            ax2.set_ylim(*cfg["spectrum_ylim"])
            ax2.set_xlabel(cfg["spectrum_xlabel"])
            ax2.set_ylabel(cfg["spectrum_ylabel"])
            ax2.set_title(cfg["spectrum_title"].format(**fmt))
            ax2.grid(alpha=0.25)

            # ---- figure-level suptitle (spans both panels) --------------
            fig.suptitle(
                cfg["suptitle"].format(**fmt),
                fontsize=12, fontweight="bold",
            )
            fig.tight_layout()

            frames.append(_frame_to_pil(fig))
            plt.close(fig)

    # GIF playback speed (ms per frame) -- raise to slow the animation down.
    _save_gif(frames, out, duration_ms=80)
    return out


if __name__ == "__main__":
    print(f"Generating Stark-splitting GIFs into {GIF_DIR}")
    for builder in (gif_ac_stark_splitting, gif_dc_stark_splitting):
        out = builder()
        kb = out.stat().st_size / 1024
        print(f"  wrote {out.name:40s} ({kb:7.1f} KB)")
    print("done")
