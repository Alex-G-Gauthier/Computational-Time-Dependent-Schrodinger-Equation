"""Expectation values, norms, and scattering coefficients.

All routines work on a single snapshot ``psi(x)`` (1-D complex array)
or on a stack of snapshots ``psi[k, :]``.  Inputs and outputs are in SI
units; the visualisation layer does the conversion to display units.
"""

from __future__ import annotations

import numpy as np

from . import units
from .grid import Grid1D


def norm(psi: np.ndarray, dx: float) -> np.ndarray:
    """``int |psi|^2 dx`` for a single snapshot or a stack."""
    return np.sum(np.abs(psi) ** 2, axis=-1) * dx


def position_expectation(psi: np.ndarray, x: np.ndarray, dx: float) -> np.ndarray:
    """``<x> = int x |psi|^2 dx``."""
    return np.sum(np.abs(psi) ** 2 * x, axis=-1) * dx


def position_variance(psi: np.ndarray, x: np.ndarray, dx: float) -> np.ndarray:
    rho = np.abs(psi) ** 2
    mean = np.sum(rho * x, axis=-1) * dx
    sq = np.sum(rho * x ** 2, axis=-1) * dx
    return sq - mean ** 2


def momentum_expectation(psi: np.ndarray, dx: float) -> np.ndarray:
    """``<p> = -i hbar int psi^* d psi/dx dx``  (central difference)."""
    if psi.ndim == 1:
        psi = psi[None, :]
        single = True
    else:
        single = False
    dpsi = np.zeros_like(psi)
    dpsi[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dx)
    p = np.real(-1j * units.HBAR
                * np.sum(np.conj(psi) * dpsi, axis=-1) * dx)
    return p[0] if single else p


def kinetic_energy(psi: np.ndarray, dx: float, m: float = units.M_E) -> np.ndarray:
    """``<T> = -hbar^2/(2m) int psi^* d^2psi/dx^2 dx``."""
    if psi.ndim == 1:
        psi = psi[None, :]
        single = True
    else:
        single = False
    lap = np.zeros_like(psi)
    lap[:, 1:-1] = (psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / dx ** 2
    T = np.real(-(units.HBAR ** 2) / (2.0 * m)
                * np.sum(np.conj(psi) * lap, axis=-1) * dx)
    return T[0] if single else T


def potential_energy(psi: np.ndarray, V: np.ndarray, dx: float) -> np.ndarray:
    """``<V> = int V |psi|^2 dx``."""
    return np.sum(V * np.abs(psi) ** 2, axis=-1) * dx


def total_energy(psi: np.ndarray, V: np.ndarray, dx: float,
                 m: float = units.M_E) -> np.ndarray:
    return kinetic_energy(psi, dx, m) + potential_energy(psi, V, dx)


def reflection_transmission(psi: np.ndarray, x: np.ndarray, dx: float,
                            barrier_left: float, barrier_right: float):
    """Estimate R, T from the integrated ``|psi|^2`` after scattering.

    ``R = int_{x<barrier_left} |psi|^2 dx``,
    ``T = int_{x>barrier_right} |psi|^2 dx``.

    Caller is responsible for choosing a snapshot at which the packet has
    fully cleared the barrier and not yet been recurrent through periodic
    boundaries; with absorbing BCs that constraint is relaxed.
    """
    rho = np.abs(psi) ** 2
    R = float(np.sum(rho[x < barrier_left]) * dx)
    T = float(np.sum(rho[x > barrier_right]) * dx)
    return R, T


def reflection_transmission_timeseries(result, barrier_left: float,
                                       barrier_right: float):
    """Return ``(R(t), T(t))`` evaluated on every saved snapshot."""
    Rs, Ts = [], []
    x = result.grid.x
    dx = result.grid.dx
    for psi in result.psi:
        R, T = reflection_transmission(psi, x, dx, barrier_left, barrier_right)
        Rs.append(R); Ts.append(T)
    return np.asarray(Rs), np.asarray(Ts)


def probability_current(psi: np.ndarray, dx: float, index: int,
                        m: float = units.M_E) -> np.ndarray:
    """Probability current ``j = (hbar/m) Im(psi* dpsi/dx)`` at grid point ``index``
    for a stack of snapshots ``psi[k, :]`` (centred difference)."""
    dpsi = (psi[..., index + 1] - psi[..., index - 1]) / (2.0 * dx)
    return units.HBAR / m * np.imag(np.conj(psi[..., index]) * dpsi)


def transmitted_probability(result, x_probe: float, m: float = units.M_E) -> float:
    """Total probability that flows past ``x_probe`` during the run,
    ``T = integral of j(x_probe, t) dt``.

    Counting the flow as it passes the probe means an absorbing edge further
    out cannot remove transmitted probability before it is counted.  Save
    enough snapshots (``n_save`` ~ 1000+) to resolve the packet passing by.
    """
    g = result.grid
    i = int(np.searchsorted(g.x, x_probe))
    j = probability_current(result.psi, g.dx, i, m=m)
    t = result.times
    return float(np.sum(0.5 * (j[1:] + j[:-1]) * np.diff(t)))   # trapezoid rule


# ----------------------------------------------------------------------------
# Convenience: timeseries of observables from a SolverResult
# ----------------------------------------------------------------------------

def timeseries(result, V_static: np.ndarray | None = None,
               m: float = units.M_E):
    """Return a dictionary of all the standard observables vs *snapshot* time.

    Keys
    ----
    ``t``        : (n_save,) snapshot times in seconds
    ``norm``     : :math:`\\int|\\psi|^2 dx` per snapshot
    ``x_mean``   : :math:`\\langle x\\rangle`
    ``x_var``    : variance of :math:`x`
    ``p_mean``   : :math:`\\langle p\\rangle`
    ``T_mean``   : :math:`\\langle\\hat T\\rangle`
    ``V_mean``   : :math:`\\langle\\hat V\\rangle`  (only if ``V_static`` given)
    ``E_mean``   : :math:`\\langle\\hat H\\rangle = T+V`     (idem)
    """
    g = result.grid
    dx = g.dx; x = g.x
    out = {
        "t":      result.times,
        "norm":   norm(result.psi, dx),
        "x_mean": position_expectation(result.psi, x, dx),
        "x_var":  position_variance(result.psi, x, dx),
        "p_mean": momentum_expectation(result.psi, dx),
        "T_mean": kinetic_energy(result.psi, dx, m=m),
    }
    if V_static is not None:
        out["V_mean"] = potential_energy(result.psi, V_static, dx)
        out["E_mean"] = out["T_mean"] + out["V_mean"]
    return out
