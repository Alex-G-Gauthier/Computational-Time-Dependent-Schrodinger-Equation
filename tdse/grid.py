"""1D spatial grid and time-grid helpers.

The simulation domain is ``x in [x_min, x_max]`` with ``N`` equally spaced
points.  We always store the *cell-centred* grid (so ``x[0] = x_min`` and
``x[-1] = x_max``); index ``i`` runs ``0..N-1`` and the interior runs
``1..N-2``.  All boundary handling in :mod:`tdse.boundaries` is built for
this convention.

Time is stored as a uniform array ``t = [0, dt, 2 dt, ..., n_t * dt]``.

For the Visscher leapfrog scheme the *time grid for the imaginary part* is
shifted by ``dt/2`` relative to the real-part grid; helpers below return
both grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import units


@dataclass(frozen=True)
class Grid1D:
    """Equally spaced 1-D spatial grid in SI units (metres)."""

    x_min: float
    x_max: float
    N: int

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.N - 1)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.N)

    @property
    def L(self) -> float:
        return self.x_max - self.x_min

    def __repr__(self) -> str:
        return (
            f"Grid1D(x_min={self.x_min:.3e} m, x_max={self.x_max:.3e} m, "
            f"N={self.N}, dx={self.dx:.3e} m)"
        )


def make_grid_nm(x_min_nm: float, x_max_nm: float, N: int) -> Grid1D:
    """Convenience: build a grid given limits in nanometres."""
    return Grid1D(x_min_nm * units.NM, x_max_nm * units.NM, N)


# --- Stability ---------------------------------------------------------------

def visscher_dt_max(grid: Grid1D, V_max: float, m: float = units.M_E,
                    safety: float = 0.9) -> float:
    """Maximum stable timestep for the Visscher leapfrog scheme.

    Von Neumann stability analysis (Part I.3 of the final project) gives the bound

    .. math::
        \\Delta t \\le \\frac{\\hbar}{\\tfrac12 V_{\\max} + 2\\hbar^2/(m\\,dx^2)}.

    We multiply by a ``safety`` factor < 1 to stay clear of the marginal
    case (which is neutrally stable and accumulates roundoff over long
    runs).
    """
    T = units.kinetic_energy_scale(grid.dx, m)  # hbar^2/(2 m dx^2)
    return safety * units.HBAR / (0.5 * V_max + 2.0 * T)


def make_time_axes(dt: float, n_steps: int):
    """Return ``(t_R, t_I)`` arrays for the staggered leapfrog scheme.

    ``t_R[k] = k*dt`` is the time at which the real part is defined.
    ``t_I[k] = (k + 1/2)*dt`` is the time at which the imaginary part is
    defined, so ``t_I`` has the *same* length as ``t_R``.
    """
    t_R = np.arange(n_steps + 1) * dt
    t_I = (np.arange(n_steps + 1) + 0.5) * dt
    return t_R, t_I


# --- Initial state factories -------------------------------------------------

def gaussian_packet(grid: Grid1D, x0: float, sigma: float,
                    k0: float = 0.0) -> np.ndarray:
    """Normalised minimum-uncertainty Gaussian wave packet.

    ``psi(x,0) = (2*pi*sigma^2)^{-1/4} exp(-(x-x0)^2/(4 sigma^2)) exp(i k0 x)``

    Parameters
    ----------
    grid   : Grid1D      simulation grid
    x0     : float       centre position (m)
    sigma  : float       position-space standard deviation (m)
    k0     : float       central wavenumber (1/m)
    """
    x = grid.x
    norm = (2.0 * np.pi * sigma ** 2) ** -0.25
    env = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2))
    psi = norm * env * np.exp(1j * k0 * x)
    return _renormalise(psi, grid.dx)


def superposition(grid: Grid1D, components):
    """Linear combination of (coefficient, psi) pairs, renormalised."""
    psi = np.zeros(grid.N, dtype=np.complex128)
    for c, phi in components:
        psi = psi + c * phi
    return _renormalise(psi, grid.dx)


def _renormalise(psi: np.ndarray, dx: float) -> np.ndarray:
    norm2 = np.sum(np.abs(psi) ** 2) * dx
    return psi / np.sqrt(norm2)
