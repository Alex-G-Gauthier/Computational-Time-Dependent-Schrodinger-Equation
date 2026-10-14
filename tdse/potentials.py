"""Library of static and time-dependent potentials.

Every factory returns one of two things:

* a NumPy array ``V(x)`` of length ``grid.N``                         (static),
* a callable ``V(x, t)`` returning a length-``N`` array              (time-dependent).

The :class:`Potential` wrapper exposes a uniform ``__call__(x, t)``
interface so the solvers do not need to special-case the static case.

All energies are in joules and all positions in metres unless explicitly
documented otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from . import units
from .grid import Grid1D


ArrayOrCallable = Union[np.ndarray, Callable[[np.ndarray, float], np.ndarray]]


@dataclass
class Potential:
    """Uniform wrapper around a static array or a ``V(x,t)`` callable."""

    V: ArrayOrCallable
    name: str = ""
    is_time_dependent: bool = False

    def __call__(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        if self.is_time_dependent:
            return self.V(x, t)  # type: ignore[misc]
        return self.V  # type: ignore[return-value]

    def max_abs(self, x: np.ndarray, t_grid: np.ndarray | None = None) -> float:
        """Worst-case |V| over the (sampled) domain, used for stability bounds."""
        if not self.is_time_dependent:
            return float(np.max(np.abs(self.V)))  # type: ignore[arg-type]
        if t_grid is None:
            t_grid = np.array([0.0])
        return max(float(np.max(np.abs(self.V(x, t)))) for t in t_grid)


# ----------------------------------------------------------------------------
# Static potentials
# ----------------------------------------------------------------------------

def free(grid: Grid1D) -> Potential:
    return Potential(V=np.zeros(grid.N), name="free particle")


def infinite_square_well(grid: Grid1D, well_left: float, well_right: float,
                         V_wall: float = 50.0 * units.EV) -> Potential:
    """Hard-wall well represented as a flat-bottom potential outside.

    ``V_wall`` must be much larger than any energy in the simulation
    *and* should not be so large that it forces an impossibly small
    timestep through the leapfrog stability bound
    ``dt <= hbar/(V_wall/2 + 2T)``.

    For a ~1 eV-scale ISW problem the default 50 eV gives a 50× safety
    margin while keeping ``dt`` only modestly constrained.

    Note: an *exact* infinite well can also be obtained "for free" by
    using hard-wall boundary conditions at grid endpoints that coincide
    with the well boundaries -- see the discussion in notebook 01.
    """
    x = grid.x
    V = np.where((x >= well_left) & (x <= well_right), 0.0, V_wall)
    return Potential(V=V, name=f"infinite well [{well_left*1e9:.2f}, {well_right*1e9:.2f}] nm")


def finite_square_well(grid: Grid1D, well_left: float, well_right: float,
                       depth: float) -> Potential:
    """Finite-depth attractive well: V = 0 outside, V = -depth inside."""
    x = grid.x
    V = np.where((x >= well_left) & (x <= well_right), -depth, 0.0)
    return Potential(V=V, name=f"finite well depth={depth/units.EV:.2f} eV")


def harmonic(grid: Grid1D, omega: float, x0: float = 0.0,
             m: float = units.M_E) -> Potential:
    """Harmonic oscillator V(x) = (1/2) m omega^2 (x - x0)^2."""
    x = grid.x
    V = 0.5 * m * omega ** 2 * (x - x0) ** 2
    return Potential(V=V, name=f"SHO omega={omega:.3e} rad/s")


def step(grid: Grid1D, x_step: float, V0: float) -> Potential:
    """Step potential: 0 for x<x_step, V0 for x>=x_step."""
    x = grid.x
    V = np.where(x >= x_step, V0, 0.0)
    return Potential(V=V, name=f"step V0={V0/units.EV:.2f} eV")


def rectangular_barrier(grid: Grid1D, x_left: float, x_right: float,
                        V0: float) -> Potential:
    """Single rectangular barrier of height V0 between x_left and x_right."""
    x = grid.x
    V = np.where((x >= x_left) & (x <= x_right), V0, 0.0)
    return Potential(V=V,
                     name=f"barrier V0={V0/units.EV:.2f} eV, w={(x_right-x_left)*1e9:.2f} nm")


def double_barrier(grid: Grid1D, x1: float, x2: float, x3: float, x4: float,
                   V0: float) -> Potential:
    """Two rectangular barriers ``[x1,x2]`` and ``[x3,x4]`` with quantum well in between."""
    x = grid.x
    V = np.zeros_like(x)
    V = np.where((x >= x1) & (x <= x2), V0, V)
    V = np.where((x >= x3) & (x <= x4), V0, V)
    return Potential(V=V,
                     name=f"double barrier V0={V0/units.EV:.2f} eV, well={(x3-x2)*1e9:.2f} nm")


def double_well(grid: Grid1D, omega: float, barrier_x: float = 0.0,
                barrier_height: float = 1.0 * units.EV, barrier_width: float = 0.05e-9,
                m: float = units.M_E) -> Potential:
    """Symmetric double well: harmonic + Gaussian bump in the middle.

    Built as ``V(x) = (1/2) m omega^2 x^2 + V_b * exp(-(x-x_b)^2 / (2 w^2))``
    so the two minima are easy to predict and the analytic eigenvalue
    splitting is well-defined.
    """
    x = grid.x
    V = (0.5 * m * omega ** 2 * x ** 2
         + barrier_height * np.exp(-((x - barrier_x) ** 2) / (2.0 * barrier_width ** 2)))
    return Potential(V=V,
                     name=f"double well, V_b={barrier_height/units.EV:.2f} eV")


# ----------------------------------------------------------------------------
# Time-dependent potentials
# ----------------------------------------------------------------------------

def driven_harmonic(omega: float, F0: float, omega_drive: float,
                    x0: float = 0.0, m: float = units.M_E) -> Potential:
    """V(x,t) = (1/2) m omega^2 (x-x0)^2 - F0 cos(omega_drive t) (x-x0)."""
    def V(x: np.ndarray, t: float) -> np.ndarray:
        return 0.5 * m * omega ** 2 * (x - x0) ** 2 - F0 * np.cos(omega_drive * t) * (x - x0)

    return Potential(V=V, is_time_dependent=True,
                     name=f"driven SHO Omega={omega_drive:.3e} rad/s")


def gaussian_pulse(omega0: float, F0: float, omega_drive: float,
                   t_centre: float, t_width: float,
                   m: float = units.M_E) -> Potential:
    """Harmonic oscillator with a Gaussian-envelope dipole pulse."""
    def V(x: np.ndarray, t: float) -> np.ndarray:
        env = np.exp(-((t - t_centre) ** 2) / (2.0 * t_width ** 2))
        return 0.5 * m * omega0 ** 2 * x ** 2 - F0 * env * np.cos(omega_drive * t) * x

    return Potential(V=V, is_time_dependent=True,
                     name=f"pulsed SHO Omega={omega_drive:.3e} rad/s")


# ----------------------------------------------------------------------------
# Landau-Zener: two Gaussian wells with a linearly swept relative depth
# ----------------------------------------------------------------------------

def landau_zener_wells(grid: Grid1D, x_L: float, x_R: float, sigma: float,
                       depth0: float, sweep_rate: float,
                       m: float = units.M_E) -> Potential:
    r"""Pair of Gaussian wells with a linearly time-swept site-energy bias.

    Static part -- two equal-depth Gaussian wells at :math:`x_L` and :math:`x_R`:

    .. math::
        V_{\rm sym}(x) = -d_0\,\big[G_L(x) + G_R(x)\big], \qquad
        G_{L,R}(x) = \exp\!\big[-(x - x_{L,R})^2/(2\sigma^2)\big].

    Bias sweep -- antisymmetric detuning that lifts the left well by
    :math:`+\alpha t/2` and lowers the right well by :math:`-\alpha t/2`:

    .. math::
        V_{\rm bias}(x,t) = \tfrac12 \alpha t \,\big[G_L(x) - G_R(x)\big].

    At :math:`t = 0` the wells are degenerate and the eigenstates form a
    symmetric / antisymmetric pair split by twice the tunnelling amplitude
    :math:`\Delta` (the minimum gap).  Far from :math:`t = 0` the
    eigenstates localise in the lower well.  Analytically,

    .. math::
        P_{\rm diabatic} = \exp\!\Big[-\frac{2\pi \Delta^2}{\hbar\,\alpha}\Big],

    the Landau-Zener formula.
    """
    def V(x_arr: np.ndarray, t: float) -> np.ndarray:
        GL = np.exp(-((x_arr - x_L) ** 2) / (2.0 * sigma ** 2))
        GR = np.exp(-((x_arr - x_R) ** 2) / (2.0 * sigma ** 2))
        return -depth0 * (GL + GR) + 0.5 * sweep_rate * t * (GL - GR)

    return Potential(V=V, is_time_dependent=True,
                     name=f"LZ wells d0={depth0/units.EV:.2f} eV, "
                          f"alpha={sweep_rate/units.EV*units.FS:.3e} eV/fs")


# ----------------------------------------------------------------------------
# Widening finite well -- for adiabatic vs sudden comparison
# ----------------------------------------------------------------------------

def widening_well(grid: Grid1D, L0: float, L1: float, t_switch: float,
                  t_smooth: float, V_wall: float = 50.0 * units.EV) -> Potential:
    r"""Finite-depth well whose half-width tanh-smoothly transitions L0 -> L1.

    .. math::
        L(t) = L_0 + (L_1 - L_0)\cdot \tfrac12\big[1 + \tanh((t - t_s)/\tau)\big],

    with :math:`V(x,t) = 0` for :math:`|x| \le L(t)/2` and
    :math:`V(x,t) = V_{\rm wall}` otherwise.  Taking
    :math:`\tau \to 0` recovers the sudden approximation; large
    :math:`\tau \gg \hbar / \Delta E` gives the adiabatic limit.
    """
    def V(x_arr: np.ndarray, t: float) -> np.ndarray:
        L_t = L0 + (L1 - L0) * 0.5 * (1.0 + np.tanh((t - t_switch) / t_smooth))
        return np.where(np.abs(x_arr) <= 0.5 * L_t, 0.0, V_wall)

    return Potential(V=V, is_time_dependent=True,
                     name=f"widening well L:{L0*1e9:.1f}->{L1*1e9:.1f} nm")


# ----------------------------------------------------------------------------
# Soft-Coulomb 1D atom -- the workhorse for photoionisation / HHG
# ----------------------------------------------------------------------------

def soft_coulomb(grid: Grid1D, Z: float = 1.0, a_soft: float = 0.1 * units.NM,
                 m: float = units.M_E) -> Potential:
    r"""1D soft-Coulomb atomic potential,
    :math:`V(x) = -Z e^2 / (4\pi\epsilon_0\sqrt{x^2 + a^2})`.

    The softening length ``a_soft`` is there to regularise the :math:`1/x`
    singularity on the grid.  With ``Z = 1, a_soft ~ 0.1 nm`` the ground
    state energy is around ``-0.5`` to ``-1 eV`` depending on the exact
    choice.
    """
    x = grid.x
    k_coul = units.E_CHARGE ** 2 / (4.0 * np.pi * 8.8541878128e-12)  # e^2/(4pi e0) [J m]
    V = -Z * k_coul / np.sqrt(x ** 2 + a_soft ** 2)
    return Potential(V=V,
                     name=f"soft-Coulomb Z={Z:.1f}, a={a_soft*1e9:.3f} nm")


def pulsed_dipole(V_static: np.ndarray, F0: float, omega_drive: float,
                  t_centre: float, t_width: float,
                  envelope: str = "gaussian",
                  n_cycles: Optional[float] = None,
                  phase: float = 0.0) -> Potential:
    r"""Wrap *any* static :math:`V_0(x)` with a dipole coupling to a pulsed field:

    .. math::
        V(x,t) = V_0(x) - F_0\,\mathrm{env}(t)\,\cos(\Omega t + \varphi)\, x.

    ``envelope``: ``"gaussian"`` gives
    :math:`\mathrm{env}(t) = \exp\!\big[-(t - t_c)^2/(2\tau^2)\big]`,
    while ``"sin2"`` gives a smooth ``sin^2`` envelope over
    ``n_cycles`` drive cycles centred at ``t_centre`` (typical of
    few-cycle-pulse simulations in strong-field AMO).
    """
    V0 = np.asarray(V_static, dtype=float)

    if envelope == "gaussian":
        def env_fn(t: float) -> float:
            return float(np.exp(-((t - t_centre) ** 2) / (2.0 * t_width ** 2)))
    elif envelope == "sin2":
        if n_cycles is None:
            raise ValueError("sin2 envelope requires n_cycles")
        T = 2.0 * np.pi / omega_drive
        t_total = n_cycles * T
        t_start = t_centre - 0.5 * t_total
        t_stop = t_centre + 0.5 * t_total

        def env_fn(t: float) -> float:
            if t <= t_start or t >= t_stop:
                return 0.0
            return float(np.sin(np.pi * (t - t_start) / t_total) ** 2)
    else:
        raise ValueError(f"unknown envelope {envelope!r}")

    def V(x: np.ndarray, t: float) -> np.ndarray:
        return V0 - F0 * env_fn(t) * np.cos(omega_drive * t + phase) * x

    return Potential(V=V, is_time_dependent=True,
                     name=f"pulsed dipole on V0, Omega={omega_drive:.3e} rad/s")
