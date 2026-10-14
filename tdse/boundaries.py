"""Boundary-condition support for the 1D TDSE.

Three BCs are implemented:

* ``hard``      :   psi(x_min) = psi(x_max) = 0       (Dirichlet)
* ``periodic``  :   psi(x_min) = psi(x_max)           (wraps the Laplacian)
* ``absorbing`` :   hard wall + complex absorbing potential (CAP) of the form
                   ``V_abs(x) = -i * eta * f(x)`` smoothly turning on near the
                   edges; this damps the wavefunction inside a *sponge layer*
                   and emulates an open boundary.

Each BC is exposed in two forms:

1. A pre-computed ``BoundaryMask`` that is multiplied against ``psi`` after
   each timestep (this implements both the hard-wall projection and any
   exponential absorbing damping).  This works for the *Visscher leapfrog*
   and *FTCS* solvers, which evolve psi by direct stencil application.
2. A modification of the discrete Laplacian operator used by the
   Crank-Nicolson solver -- see :func:`build_laplacian`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse

from . import units
from .grid import Grid1D


BCName = Literal["hard", "periodic", "absorbing"]


@dataclass
class BoundaryMask:
    """Per-step multiplicative mask for psi, plus an absorbing imaginary V."""

    mask: np.ndarray              # real, in [0,1], applied as psi *= mask
    V_imag: np.ndarray            # imag part of an effective potential (J)
    name: str

    def apply(self, psi: np.ndarray) -> np.ndarray:
        return psi * self.mask


def hard_wall_mask(grid: Grid1D) -> BoundaryMask:
    mask = np.ones(grid.N)
    mask[0] = 0.0
    mask[-1] = 0.0
    return BoundaryMask(mask=mask, V_imag=np.zeros(grid.N), name="hard wall")


def periodic_mask(grid: Grid1D) -> BoundaryMask:
    """Periodic BCs need no mask -- they are handled in the Laplacian / stencil.

    Returning all-ones means no damping is applied; the solver detects
    ``name == 'periodic'`` and uses the periodic stencil.
    """
    return BoundaryMask(mask=np.ones(grid.N),
                        V_imag=np.zeros(grid.N), name="periodic")


def absorbing_mask(grid: Grid1D, sponge_width: float,
                   eta: float | None = None,
                   profile: str = "cos2",
                   m: float = units.M_E) -> BoundaryMask:
    """Complex-absorbing-potential sponge layer.

    A thin layer of width ``sponge_width`` (m) at each edge is given a
    smoothly rising imaginary potential ``-i * eta * f(s)`` where
    ``s in [0,1]`` runs from inside-edge of the sponge to the wall.  We
    implement this as a *multiplicative damping mask* per timestep,

    .. math::
        \\text{mask}(x) = \\exp\\big(-\\,\\eta\\,\\Delta t \\, f(s) / \\hbar\\big),

    constructed by the solver from ``V_imag`` and the timestep, plus a
    hard wall at the very edge to suppress any leakage.  We expose only
    the imaginary part of V here; the actual exponential mask is built by
    :func:`make_absorbing_step_mask` once ``dt`` is known.

    ``eta`` defaults to a value that yields O(0.99) absorption per
    traversal of the sponge for an electron at 1 eV.
    """
    x = grid.x
    L_left = grid.x_min
    L_right = grid.x_max

    if eta is None:
        # rough scale: hbar / sponge_traversal_time at v ~ sqrt(2 E/m), E=1 eV
        v_typ = np.sqrt(2.0 * 1.0 * units.EV / m)
        tau = sponge_width / v_typ
        eta = 5.0 * units.HBAR / tau  # ~5 e-foldings across the sponge

    s_left = np.clip((L_left + sponge_width - x) / sponge_width, 0.0, 1.0)
    s_right = np.clip((x - (L_right - sponge_width)) / sponge_width, 0.0, 1.0)

    if profile == "cos2":
        f_left = np.where(s_left > 0, np.sin(0.5 * np.pi * s_left) ** 2, 0.0)
        f_right = np.where(s_right > 0, np.sin(0.5 * np.pi * s_right) ** 2, 0.0)
    elif profile == "linear":
        f_left = s_left
        f_right = s_right
    else:
        raise ValueError(f"unknown profile {profile!r}")

    V_imag = -eta * (f_left + f_right)

    mask = np.ones(grid.N)
    mask[0] = 0.0
    mask[-1] = 0.0

    return BoundaryMask(mask=mask, V_imag=V_imag,
                        name=f"absorbing(width={sponge_width*1e9:.1f} nm)")


def make_absorbing_step_mask(bm: BoundaryMask, dt: float) -> np.ndarray:
    """Convert (V_imag, dt) into the per-step exponential damping factor."""
    return bm.mask * np.exp(bm.V_imag * dt / units.HBAR)


# ----------------------------------------------------------------------------
# Laplacian operator with BC -- used by Crank-Nicolson
# ----------------------------------------------------------------------------

def build_laplacian(grid: Grid1D, bc: BCName) -> sparse.csr_matrix:
    """Second-order central-difference Laplacian on the grid.

    Returns a sparse ``(N, N)`` matrix ``D2`` such that
    ``(D2 @ psi)[i] ~ (psi[i+1] - 2 psi[i] + psi[i-1]) / dx^2``.

    The interior rows are identical for all BCs; only the first and last
    rows change.
    """
    N, dx = grid.N, grid.dx
    main = -2.0 * np.ones(N)
    off = np.ones(N - 1)
    D2 = sparse.diags([off, main, off], offsets=[-1, 0, 1], format="lil")

    if bc == "hard":
        # Dirichlet: psi(0) = psi(N-1) = 0.  In matrix form we keep
        # the standard rows, but downstream solvers will set those entries
        # of psi to zero (or we project).  For Crank-Nicolson we build the
        # interior-only matrix later, so here we just leave it.
        pass
    elif bc == "periodic":
        D2[0, -1] = 1.0
        D2[-1, 0] = 1.0
    elif bc == "absorbing":
        # The CAP itself is added to V; the Laplacian still uses Dirichlet.
        pass
    else:
        raise ValueError(f"unknown BC {bc!r}")

    return (D2 / dx ** 2).tocsr()
