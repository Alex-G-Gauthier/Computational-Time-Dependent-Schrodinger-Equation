"""Potentials used in the three simulations: free space, the harmonic
oscillator and a rectangular barrier.

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


def harmonic(grid: Grid1D, omega: float, x0: float = 0.0,
             m: float = units.M_E) -> Potential:
    """Harmonic oscillator V(x) = (1/2) m omega^2 (x - x0)^2."""
    x = grid.x
    V = 0.5 * m * omega ** 2 * (x - x0) ** 2
    return Potential(V=V, name=f"SHO omega={omega:.3e} rad/s")


def rectangular_barrier(grid: Grid1D, x_left: float, x_right: float,
                        V0: float) -> Potential:
    """Single rectangular barrier of height V0 between x_left and x_right."""
    x = grid.x
    V = np.where((x >= x_left) & (x <= x_right), V0, 0.0)
    return Potential(V=V,
                     name=f"barrier V0={V0/units.EV:.2f} eV, w={(x_right-x_left)*1e9:.2f} nm")
