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


def project_onto(psi: np.ndarray, phi: np.ndarray, dx: float) -> complex:
    """``<phi|psi> = int phi^* psi dx``."""
    return complex(np.sum(np.conj(phi) * psi) * dx)


# ----------------------------------------------------------------------------
# Spectral / projection observables
# ----------------------------------------------------------------------------

def populations(result, states: np.ndarray) -> np.ndarray:
    """Populations :math:`P_n(t) = |\\langle \\phi_n | \\psi(t)\\rangle|^2`.

    Parameters
    ----------
    result : SolverResult
    states : (n_states, N) array of reference states, normalised in the
             same convention as ``result.psi`` (``int |phi|^2 dx = 1``).

    Returns
    -------
    P : (n_save, n_states) real array
    """
    dx = result.grid.dx
    # overlap matrix = (n_save, N) @ (N, n_states)
    overlap = result.psi @ (np.conj(states).T) * dx
    return np.abs(overlap) ** 2


def autocorrelation(result, psi0: np.ndarray) -> np.ndarray:
    """Autocorrelation :math:`C(t) = \\langle \\psi(0) | \\psi(t) \\rangle`
    evaluated on the saved snapshots.  Complex-valued.
    """
    dx = result.grid.dx
    return (result.psi @ np.conj(psi0)) * dx


def bound_population(result, states: np.ndarray) -> np.ndarray:
    """Total population in the listed reference states:
    :math:`P_{\\rm bound}(t) = \\sum_n |\\langle\\phi_n|\\psi(t)\\rangle|^2`.

    Useful as an *ionisation* diagnostic when combined with absorbing BCs:
    whatever leaves the bound subspace either sits in the continuum or has
    been absorbed at the edges.
    """
    return np.sum(populations(result, states), axis=-1)


def dipole_acceleration(result, V_grad: np.ndarray,
                        field_fn=None, q: float = -units.E_CHARGE,
                        m: float = units.M_E) -> np.ndarray:
    r"""Dipole acceleration via Ehrenfest,

    .. math::
        a(t) = \frac{1}{m}\,\Big\langle -\frac{\partial V}{\partial x}
                                      + q\,E(t)\Big\rangle.

    ``V_grad`` is the spatial derivative of the *static* part of the
    potential (shape ``(N,)``), typically computed with ``np.gradient(V0, dx)``.
    ``field_fn`` is an optional callable returning the laser field
    ``E(t)`` (in V/m); if provided the uniform ``q E(t)`` contribution is
    added.  ``q`` defaults to the *electron* charge (negative).

    Returns a real array of shape ``(n_save,)`` in m/s^2.  This is the
    standard input to HHG spectra:
    :math:`S(\omega) \propto |\tilde a(\omega)|^2`.
    """
    dx = result.grid.dx
    rho = np.abs(result.psi) ** 2           # (n_save, N)
    a = -np.sum(rho * V_grad, axis=-1) * dx / m
    if field_fn is not None:
        a = a + q * np.asarray([field_fn(t) for t in result.times]) / m
    return a


def power_spectrum(signal: np.ndarray, dt: float, window: str = "hann"):
    r"""One-sided power spectrum :math:`|\tilde s(\omega)|^2` of a uniformly
    sampled real or complex signal.

    Returns ``(omega, S)`` with ``omega`` in rad/s and ``S`` the squared
    magnitude of the windowed Fourier transform.  A Hann window is applied
    by default to suppress ringing from finite record length.
    """
    n = len(signal)
    if window == "hann":
        w = np.hanning(n)
    elif window == "none" or window is None:
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window {window!r}")
    s = np.asarray(signal) * w
    # rfft if real, fft otherwise
    if np.iscomplexobj(s):
        Sk = np.fft.fftshift(np.fft.fft(s) * dt)
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
    else:
        Sk = np.fft.rfft(s) * dt
        freqs = np.fft.rfftfreq(n, d=dt)
    omega = 2.0 * np.pi * freqs
    return omega, np.abs(Sk) ** 2


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
