"""Closed-form benchmark solutions used to validate the numerical solvers.

* :func:`free_gaussian` -- exact time evolution of a Gaussian wave packet
  in free space (Schiff, *Quantum Mechanics*, Eq. 3.13).

* :func:`sho_eigenstate`, :func:`sho_coherent_state` -- harmonic oscillator
  eigenfunctions (physicist's Hermite polynomials) and the coherent state,
  a superposition of them that moves like a classical particle.

* :func:`barrier_transmission` -- analytic transmission coefficient for a
  rectangular barrier (Griffiths Eq. 2.171).

* :func:`transfer_matrix_TR` -- numerically exact transmission for any
  piecewise-constant potential, used as an independent check.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_hermite, factorial

from . import units


# ----------------------------------------------------------------------------
# Free Gaussian wave packet, exact time evolution
# ----------------------------------------------------------------------------

def free_gaussian(x: np.ndarray, t: float, x0: float, sigma0: float,
                  k0: float, m: float = units.M_E) -> np.ndarray:
    r"""Exact time evolution of a free minimum-uncertainty Gaussian.

    .. math::
        \psi(x,t) = \frac{1}{(2\pi)^{1/4}\sqrt{\sigma_0 + i\hbar t/(2 m \sigma_0)}}
                    \exp\!\left[\frac{-(x - x_0 - \hbar k_0 t/m)^2}
                                       {4\sigma_0(\sigma_0 + i\hbar t/(2 m \sigma_0))}
                                + i k_0 x - i \frac{\hbar k_0^2}{2 m} t\right].

    The wave packet's spatial r.m.s. width grows as
    :math:`\sigma(t) = \sigma_0 \sqrt{1 + (\hbar t/(2 m \sigma_0^2))^2}`.
    """
    hbar = units.HBAR
    alpha = sigma0 + 1j * hbar * t / (2.0 * m * sigma0)
    norm = (2.0 * np.pi) ** -0.25 / np.sqrt(alpha)
    xc = x0 + hbar * k0 * t / m
    phase = 1j * k0 * x - 1j * hbar * k0 ** 2 * t / (2.0 * m)
    return norm * np.exp(-(x - xc) ** 2 / (4.0 * sigma0 * alpha) + phase)


def free_gaussian_width(t: float, sigma0: float, m: float = units.M_E) -> float:
    return sigma0 * np.sqrt(1.0 + (units.HBAR * t / (2.0 * m * sigma0 ** 2)) ** 2)


# ----------------------------------------------------------------------------
# Harmonic oscillator
# ----------------------------------------------------------------------------

def sho_eigenstate(n: int, x: np.ndarray, omega: float, x0: float = 0.0,
                   m: float = units.M_E) -> np.ndarray:
    """Normalised oscillator eigenfunction phi_n.

    .. math::
        \\phi_n(x) = \\frac{1}{\\sqrt{2^n n!}}
                    \\Big(\\frac{m\\omega}{\\pi\\hbar}\\Big)^{1/4}
                    H_n(\\xi) e^{-\\xi^2/2}, \\quad
                    \\xi = \\sqrt{m\\omega/\\hbar}\\,(x-x_0).
    """
    xi = np.sqrt(m * omega / units.HBAR) * (x - x0)
    norm = 1.0 / np.sqrt(2.0 ** n * factorial(n)) \
           * (m * omega / (np.pi * units.HBAR)) ** 0.25
    return norm * eval_hermite(n, xi) * np.exp(-xi ** 2 / 2.0)


def sho_energy(n: int, omega: float) -> float:
    return units.HBAR * omega * (n + 0.5)


def sho_coherent_state(x: np.ndarray, alpha: complex, omega: float,
                       x0: float = 0.0, m: float = units.M_E) -> np.ndarray:
    """Coherent state |alpha> as a position-space Gaussian."""
    sigma0 = np.sqrt(units.HBAR / (2.0 * m * omega))
    x_disp = sigma0 * np.sqrt(2.0) * np.real(alpha)
    p_disp = np.sqrt(2.0 * m * units.HBAR * omega) * np.imag(alpha)
    k0 = p_disp / units.HBAR
    arg = (x - x0 - x_disp) ** 2 / (4.0 * sigma0 ** 2)
    norm = (2.0 * np.pi * sigma0 ** 2) ** -0.25
    return norm * np.exp(-arg) * np.exp(1j * k0 * x)


# ----------------------------------------------------------------------------
# Step potential
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Rectangular barrier
# ----------------------------------------------------------------------------

def barrier_transmission(E: float, V0: float, a: float,
                         m: float = units.M_E) -> float:
    """Analytic transmission probability for a rectangular barrier of
    height ``V0`` and width ``a``.

    Below the barrier (E < V0):
        T = 1 / [1 + V0^2 sinh^2(kappa a) / (4 E (V0 - E))]
        with kappa = sqrt(2 m (V0 - E))/hbar.
    Above the barrier (E > V0):
        T = 1 / [1 + V0^2 sin^2(k' a) / (4 E (E - V0))]
        with k' = sqrt(2 m (E - V0))/hbar.
    At E = V0:  T = 1 / (1 + m V0 a^2 / (2 hbar^2)).
    """
    hbar = units.HBAR
    if abs(E - V0) <= 1e-12 * V0:
        return 1.0 / (1.0 + m * V0 * a ** 2 / (2.0 * hbar ** 2))
    if E < V0:
        kappa = np.sqrt(2.0 * m * (V0 - E)) / hbar
        return 1.0 / (1.0 + (V0 ** 2 * np.sinh(kappa * a) ** 2)
                              / (4.0 * E * (V0 - E)))
    kp = np.sqrt(2.0 * m * (E - V0)) / hbar
    return 1.0 / (1.0 + (V0 ** 2 * np.sin(kp * a) ** 2)
                          / (4.0 * E * (E - V0)))


def barrier_T_curve(V0: float, a: float, E_min: float, E_max: float,
                    n_points: int = 400, m: float = units.M_E):
    E = np.linspace(E_min, E_max, n_points)
    T = np.array([barrier_transmission(Ei, V0, a, m) for Ei in E])
    return E, T


# ----------------------------------------------------------------------------
# Transfer-matrix method for *arbitrary* 1D piecewise-constant potentials
# ----------------------------------------------------------------------------

def transfer_matrix_TR(E: float, x: np.ndarray, V: np.ndarray,
                       m: float = units.M_E):
    """Exact transmission and reflection coefficients for an arbitrary 1D
    potential ``V(x)`` sampled on the grid ``x``.

    The potential is approximated as a chain of *piecewise-constant* slabs,
    one per grid cell.  Inside each slab the wave function is a sum of
    plane (or evanescent) waves; matching ``psi`` and ``psi'`` across each
    interface yields a 2x2 transfer matrix.  Multiplying the matrices
    produces the exact (within the piecewise-constant approximation)
    scattering amplitudes for the whole structure.

    The asymptotic regions are taken to be the leftmost and rightmost
    slabs; ``V`` should be approximately constant there.  Returns
    ``(T, R)`` such that ``T + R == 1`` to machine precision when ``V``
    is real (no absorption).

    Notes
    -----
    For ``V(x) >= E`` everywhere on the left or right asymptote the
    incoming current is ill-defined and the routine returns ``(0.0, 1.0)``.
    """
    hbar = units.HBAR
    EJ = float(E)

    # Asymptotic potentials -- assume left of x[0] and right of x[-1] flat
    V_L = float(V[0])
    V_R = float(V[-1])
    if EJ <= V_L:
        # below the left asymptote -- no propagating incident state
        return 0.0, 1.0

    def _wavenumber(E_, V_):
        # complex k so the same formula handles classically forbidden
        # regions; convention: real(k) >= 0
        kk = np.sqrt(2.0 * m * (E_ - V_) + 0j) / hbar
        if kk.real < 0.0:
            kk = -kk
        return kk

    M = np.eye(2, dtype=np.complex128)
    Ncells = len(x) - 1
    # Convention: amplitudes (A_j, B_j) in slab j are defined *relative to
    # the left edge x_j of the slab*, so psi_j(xi) = A_j e^{i k_j xi} +
    # B_j e^{-i k_j xi} for xi = x - x_j in [0, dx_j].  Propagating across
    # slab j by width dx multiplies (A,B) by diag(e^{i k_j dx}, e^{-i k_j dx}),
    # then matching at the interface x_{j+1} between slab j and slab j+1
    # gives the standard Fresnel-like relation
    #   [A_{j+1}, B_{j+1}]^T = (1/(2 k_{j+1}))
    #     [[k_{j+1}+k_j, k_{j+1}-k_j],
    #      [k_{j+1}-k_j, k_{j+1}+k_j]] [A_j', B_j']^T
    # where (A_j', B_j') = Pprop @ (A_j, B_j) are the amplitudes at the
    # *right* edge of slab j.
    for j in range(Ncells):
        kj = _wavenumber(EJ, float(V[j]))
        kjp = _wavenumber(EJ, float(V[j + 1]))
        if abs(kjp) < 1e-300:
            continue
        dx = float(x[j + 1] - x[j])

        Pprop = np.array([[np.exp(1j * kj * dx), 0.0],
                          [0.0, np.exp(-1j * kj * dx)]],
                         dtype=np.complex128)
        Tint = 0.5 / kjp * np.array(
            [[kjp + kj, kjp - kj],
             [kjp - kj, kjp + kj]], dtype=np.complex128)

        M = Tint @ Pprop @ M

    # Transmission of an incident wave from the left:
    #   left:  A e^{ikL x} + B e^{-ikL x},   right: F e^{ikR x}     (no return)
    # So [F, 0]^T = M [A, B]^T  =>  M21 A + M22 B = 0  =>  B = -M21/M22 A
    # and F = (M11 - M12 M21/M22) A.
    M11, M12 = M[0, 0], M[0, 1]
    M21, M22 = M[1, 0], M[1, 1]
    if abs(M22) < 1e-300:
        return 0.0, 1.0
    r = -M21 / M22
    t = M11 - M12 * M21 / M22

    kL = _wavenumber(EJ, V_L).real
    kR = _wavenumber(EJ, V_R).real
    # Transmission coefficient with current normalisation factor:
    if EJ <= V_R:
        # right side classically forbidden -- T=0
        T = 0.0
    else:
        T = (kR / kL) * abs(t) ** 2
    R = abs(r) ** 2
    # T + R must be 1 for a real potential; we return them as is so the
    # caller can monitor sanity.
    return float(T), float(R)


def transfer_matrix_T_curve(x: np.ndarray, V: np.ndarray,
                            E_min: float, E_max: float,
                            n_points: int = 400, m: float = units.M_E):
    """``T(E), R(E)`` over a uniform energy scan, via :func:`transfer_matrix_TR`."""
    E = np.linspace(E_min, E_max, n_points)
    T = np.zeros_like(E)
    R = np.zeros_like(E)
    for i, Ei in enumerate(E):
        T[i], R[i] = transfer_matrix_TR(float(Ei), x, V, m=m)
    return E, T, R
