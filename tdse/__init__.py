"""tdse - 1D time-dependent Schrödinger equation simulator.

Modules
-------
units        : SI constants and display-scale conversions.
grid         : 1D spatial and temporal grids.
potentials   : free space, harmonic oscillator and rectangular barrier.
boundaries   : hard-wall, periodic, and absorbing boundary helpers.
solvers      : Visscher leapfrog, naive FTCS, Crank-Nicolson, and FFT split-operator.
observables  : norms, expectation values, transmission/reflection.
analytical   : closed-form reference solutions.
viz          : animation and space-time heatmap utilities.
"""

from . import units, grid, potentials, boundaries, solvers, observables, analytical, viz

__all__ = [
    "units",
    "grid",
    "potentials",
    "boundaries",
    "solvers",
    "observables",
    "analytical",
    "viz",
]
