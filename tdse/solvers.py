"""Four time-integrators for the 1-D time-dependent Schrödinger equation,
plus an imaginary-time relaxation helper for ground-state preparation.

Notation (used throughout the docstrings)
-----------------------------------------

* ``\\psi(x,t) = R(x,t) + i\\,I(x,t)``  -- real / imaginary parts of psi.
* ``H = -\\hbar^2/(2m) \\partial_x^2 + V(x,t)``       -- the Hamiltonian.
* ``\\mathcal H = (\\Delta t/\\hbar) H``               -- the *dimensionless*
  step operator.  In the discretised form,

  .. math::
      (\\mathcal H \\psi)_n = -\\frac{\\hbar\\,\\Delta t}{2 m\\,\\Delta x^2}
                              (\\psi_{n+1} - 2\\psi_n + \\psi_{n-1})
                              + \\frac{\\Delta t}{\\hbar} V_n \\,\\psi_n.

The TDSE ``i\\hbar \\partial_t \\psi = H \\psi`` is then ``i \\partial_\\tau \\psi
= \\mathcal H \\psi`` in the dimensionless time ``\\tau = t/\\Delta t``, and we
need a way to advance one step ``\\tau \\to \\tau + 1``.

Four schemes are provided:

* :class:`FTCSSolver`           -- naive ``\\psi^{n+1} = \\psi^n - i \\mathcal H \\psi^n``.
                                   Forward Euler in time.  Provably *unconditionally
                                   unstable*: every Fourier mode has a growth factor
                                   ``|1 - i \\lambda| > 1``.  Included only to demonstrate
                                   the failure mode.
* :class:`LeapfrogSolver`       -- Visscher's staggered leapfrog.  Real and imaginary
                                   parts live on time-grids offset by ``dt/2``:

                                   .. math::
                                       R^{n+1} &= R^n + \\mathcal H \\, I^{n+1/2} \\\\
                                       I^{n+3/2} &= I^{n+1/2} - \\mathcal H \\, R^{n+1}.

                                   Conditionally stable for
                                   ``\\Delta t \\le \\hbar/[\\tfrac12 V_{\\max} + 2T]``,
                                   where ``T = \\hbar^2/(2m\\Delta x^2)``.  Conserves
                                   the modified norm ``\\sum_n R_n^2 + I_n^{(-)} I_n^{(+)}``
                                   to *machine precision*.
* :class:`CrankNicolsonSolver`  -- implicit, second-order in time, exactly unitary:

                                   .. math::
                                       (1 + \\tfrac{i}{2} \\mathcal H) \\psi^{n+1}
                                       = (1 - \\tfrac{i}{2} \\mathcal H) \\psi^n.

                                   Unconditionally stable; benchmark / "ground truth"
                                   solver.  Implemented with banded sparse linear
                                   algebra so the per-step cost is O(N).
* :class:`SplitOperatorSolver` -- FFT-based Strang splitting,

                                   .. math::
                                       \\psi^{n+1} = e^{-i V \\Delta t/(2\\hbar)}\\,
                                                     \\mathcal F^{-1}\\,
                                                     e^{-i \\hbar k^2 \\Delta t/(2m)}\\,
                                                     \\mathcal F\\,
                                                     e^{-i V \\Delta t/(2\\hbar)}\\,\\psi^n,

                                   each operator in its diagonal basis (position for
                                   ``V``, momentum for ``T``).  Exactly unitary and
                                   second-order in :math:`\\Delta t` with no spatial
                                   discretisation error in ``T``; cost O(N log N) per
                                   step.  Natural fit for periodic boundaries; the
                                   hard / absorbing variants apply the same mask as the
                                   other solvers.  Provides an independent fourth
                                   benchmark for the leapfrog scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu

from . import units
from .boundaries import (BCName, BoundaryMask, build_laplacian,
                         make_absorbing_step_mask)
from .grid import Grid1D
from .potentials import Potential


# ============================================================================
# Result container
# ============================================================================

@dataclass
class SolverResult:
    """Output of a full simulation run.

    Storage policy: rather than keeping every timestep (which would use
    O(N * n_steps) memory), the solvers down-sample to ``n_save``
    snapshots evenly spaced in time.  The full diagnostic arrays
    (norm, energy, ...) are kept at every step.
    """
    grid: Grid1D
    times: np.ndarray                 # (n_save,) snapshot times in s
    psi: np.ndarray                   # (n_save, N) complex
    diag_times: np.ndarray            # (n_steps+1,) diagnostic times in s
    norm: np.ndarray                  # (n_steps+1,) integral of |psi|^2 dx
    norm_visscher: Optional[np.ndarray] = None  # only for leapfrog
    energy: Optional[np.ndarray] = None         # (n_steps+1,) <H> in J
    extras: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"SolverResult(N={self.grid.N}, "
                f"n_save={len(self.times)}, "
                f"t_final={self.diag_times[-1]:.3e} s)")


# ============================================================================
# Helpers used by all solvers
# ============================================================================

def _expectation_energy(psi: np.ndarray, V: np.ndarray, dx: float,
                        m: float) -> float:
    """<psi|H|psi> using a 3-point Laplacian and Dirichlet padding."""
    N = psi.size
    lap = np.zeros_like(psi)
    lap[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dx ** 2
    H_psi = -(units.HBAR ** 2) / (2.0 * m) * lap + V * psi
    return float(np.real(np.sum(np.conj(psi) * H_psi) * dx))


def _norm_squared(psi: np.ndarray, dx: float) -> float:
    return float(np.sum(np.abs(psi) ** 2) * dx)


def _stencil_apply_H(psi: np.ndarray, V: np.ndarray, dx: float, m: float,
                     bc: BCName) -> np.ndarray:
    """Apply H to psi using the 3-point Laplacian with the chosen BC.

    Returns ``H psi`` in joules (same units as ``V * psi``).  Boundaries
    that are not ``periodic`` use Dirichlet padding (psi=0 outside).
    """
    N = psi.size
    lap = np.empty_like(psi)
    if bc == "periodic":
        lap[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dx ** 2
        lap[0] = (psi[1] - 2.0 * psi[0] + psi[-1]) / dx ** 2
        lap[-1] = (psi[0] - 2.0 * psi[-1] + psi[-2]) / dx ** 2
    else:
        lap[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dx ** 2
        lap[0] = (psi[1] - 2.0 * psi[0]) / dx ** 2
        lap[-1] = (-2.0 * psi[-1] + psi[-2]) / dx ** 2
    return -(units.HBAR ** 2) / (2.0 * m) * lap + V * psi


# ============================================================================
# Base class
# ============================================================================

class _BaseSolver:
    """Common bookkeeping for the three time integrators."""

    name: str = "base"

    def __init__(self, grid: Grid1D, potential: Potential, dt: float,
                 m: float = units.M_E, bc: BCName = "hard",
                 boundary_mask: Optional[BoundaryMask] = None):
        self.grid = grid
        self.potential = potential
        self.dt = float(dt)
        self.m = float(m)
        self.bc = bc
        self.boundary_mask = boundary_mask

    # public API --------------------------------------------------------------
    def run(self, psi0: np.ndarray, n_steps: int, n_save: int = 200,
            track_energy: bool = True) -> SolverResult:
        raise NotImplementedError

    # internal helpers --------------------------------------------------------
    def _save_indices(self, n_steps: int, n_save: int) -> np.ndarray:
        n_save = min(n_save, n_steps + 1)
        return np.unique(np.linspace(0, n_steps, n_save).astype(int))


# ============================================================================
# 1. Naive FTCS (Forward-Time Centered-Space) -- unstable, demonstration only
# ============================================================================

class FTCSSolver(_BaseSolver):
    """Forward Euler in time, centred difference in space.

    Update rule (with ``\\mathcal H = dt/hbar * H``):

    .. math::
        \\psi^{n+1} = \\psi^n - i \\mathcal H \\psi^n.

    Von Neumann analysis shows the amplification factor for plane wave
    ``e^{ikx}`` is ``g = 1 - i\\lambda`` with ``lambda`` real, so
    ``|g| = sqrt(1 + lambda^2) > 1``.  *Every* mode grows -- the only
    question is how fast.  This solver is included to demonstrate that
    failure visually.
    """

    name = "FTCS"

    def run(self, psi0, n_steps, n_save=200, track_energy=True):
        N = self.grid.N
        dx = self.grid.dx
        psi = psi0.astype(np.complex128).copy()

        save_idx = self._save_indices(n_steps, n_save)
        save_set = set(int(i) for i in save_idx)

        psi_save = np.zeros((len(save_idx), N), dtype=np.complex128)
        t_save = np.zeros(len(save_idx))
        norm = np.zeros(n_steps + 1)
        energy = np.zeros(n_steps + 1) if track_energy else None
        diag_t = np.arange(n_steps + 1) * self.dt

        # The hard-wall and periodic BCs are enforced by the H stencil itself
        # (Dirichlet ghost-outside or wraparound).  Only absorbing BC needs a
        # multiplicative per-step damping mask.
        if self.bc == "absorbing" and self.boundary_mask is not None:
            damp = make_absorbing_step_mask(self.boundary_mask, self.dt)
        else:
            damp = None

        save_pos = 0
        for n in range(n_steps + 1):
            V = self.potential(self.grid.x, n * self.dt)
            norm[n] = _norm_squared(psi, dx)
            if track_energy:
                energy[n] = _expectation_energy(psi, V, dx, self.m)
            if n in save_set:
                psi_save[save_pos] = psi
                t_save[save_pos] = n * self.dt
                save_pos += 1
            if n == n_steps:
                break

            H_psi = _stencil_apply_H(psi, V, dx, self.m, self.bc)
            psi = psi - 1j * self.dt / units.HBAR * H_psi

            if damp is not None:
                psi = psi * damp

        return SolverResult(grid=self.grid, times=t_save, psi=psi_save,
                            diag_times=diag_t, norm=norm, energy=energy,
                            extras={"solver": self.name})


# ============================================================================
# 2. Visscher staggered leapfrog
# ============================================================================

class LeapfrogSolver(_BaseSolver):
    """Explicit staggered leapfrog ("Visscher") solver.

    Decomposing :math:`\\psi = R + i I` and discretising the TDSE on a
    grid that is *uniform in space* but *staggered in time* (R at integer
    steps :math:`n\\Delta t`, I at half-steps :math:`(n+\\tfrac12)\\Delta t`)
    yields the explicit update

    .. math::
        R^{n+1} = R^{n}   + \\frac{\\Delta t}{\\hbar}\\,H\\,I^{n+1/2}, \\\\
        I^{n+3/2} = I^{n+1/2} - \\frac{\\Delta t}{\\hbar}\\,H\\,R^{n+1}.

    To bootstrap, given :math:`\\psi(0) = R^0 + i I^0` (defined at the same
    time), we shift :math:`I` forward by half a step using one half of a
    Crank-Nicolson move (or, sufficient here, a Taylor expansion):
    :math:`I^{1/2} = I^0 - \\tfrac{1}{2}\\tfrac{\\Delta t}{\\hbar} H R^0`.
    The same trick is run *in reverse* at the end to bring R and I back
    to the same time when we write a snapshot.

    Stability bound (von Neumann):

    .. math::
        \\Delta t \\le \\frac{\\hbar}{\\tfrac12 V_{\\max} + 2 T}, \\qquad
        T = \\frac{\\hbar^2}{2 m \\Delta x^2}.

    Conserved quantity (the *Visscher norm*):

    .. math::
        \\mathcal N^{n} = \\sum_k \\Delta x \\big[(R^{n}_k)^2
                                       + I^{n-1/2}_k\\,I^{n+1/2}_k\\big].

    This is the discrete analogue of :math:`\\int|\\psi|^2 dx` and is
    conserved by the leapfrog update *exactly* (to machine precision).

    Note about the bootstrap offset.  Because the user supplies
    :math:`\\psi(0) = R^0 + i I^0` with both components defined at
    :math:`t=0`, but Visscher's scheme stores :math:`I` at half-steps, we
    bootstrap :math:`I^{\\pm 1/2} = I^0 \\mp \\tfrac{\\Delta t}{2\\hbar} H R^0`
    (Euler half-steps, both :math:`O(\\Delta t^2)`).  Substituting these
    into :math:`\\mathcal N^0` gives an exact analytic offset

    .. math::
        \\mathcal N^0 - \\!\\int|\\psi(0)|^2 dx
            = -\\Big(\\tfrac{\\Delta t}{2\\hbar}\\Big)^2
              \\sum_k (H R^0)_k^2 \\,\\Delta x
            = -\\Big(\\tfrac{\\Delta t}{2\\hbar}\\Big)^2
              \\langle R^0 | H^2 | R^0\\rangle.

    For a state normalised to 1 this offset is :math:`O(\\Delta t^2)` and
    *constant in time* (Visscher conservation is then perfect about that
    biased baseline).  Reproduced numerically in
    ``tools/_verify_visscher.py``: predicted vs. measured offset agree to
    10 decimal places for the (``phi_1+phi_2)/sqrt(2)`` ISW benchmark.
    """

    name = "leapfrog"

    def run(self, psi0, n_steps, n_save=200, track_energy=True):
        N = self.grid.N
        dx = self.grid.dx
        dt = self.dt

        R = np.real(psi0).astype(np.float64).copy()
        I0 = np.imag(psi0).astype(np.float64).copy()  # I at t = 0
        # The user supplies psi at t=0, but the Visscher scheme stores I only
        # at half-integer times.  We interpret I0 as the value at t=0 and
        # use an O(dt) forward / backward Euler half-step to obtain
        #   I^{+1/2} = I0 - (dt/(2*hbar)) * H * R^0   (forward),
        #   I^{-1/2} = I0 + (dt/(2*hbar)) * H * R^0   (backward).
        # Both have O(dt^2) error which is the same order as the scheme itself.
        V0 = self.potential(self.grid.x, 0.0)
        H_R = _stencil_apply_H(R, V0, dx, self.m, self.bc)
        I_half = I0 - 0.5 * dt / units.HBAR * H_R         # I^{+1/2}
        I_prev = I0 + 0.5 * dt / units.HBAR * H_R         # I^{-1/2}

        save_idx = self._save_indices(n_steps, n_save)
        save_set = set(int(i) for i in save_idx)
        psi_save = np.zeros((len(save_idx), N), dtype=np.complex128)
        t_save = np.zeros(len(save_idx))

        norm = np.zeros(n_steps + 1)
        norm_v = np.zeros(n_steps + 1)
        energy = np.zeros(n_steps + 1) if track_energy else None
        diag_t = np.arange(n_steps + 1) * dt

        # Hard-wall and periodic BCs are enforced by the H stencil; absorbing
        # BC additionally damps psi inside the sponge layer.
        if self.bc == "absorbing" and self.boundary_mask is not None:
            damp = make_absorbing_step_mask(self.boundary_mask, dt)
        else:
            damp = None

        # diagnostics at n=0 ------------------------------------------------
        psi_now = R + 1j * I0
        norm[0] = _norm_squared(psi_now, dx)
        # Visscher modified norm  N^0 = sum[ R^0^2 + I^{-1/2} I^{+1/2} ]
        norm_v[0] = float(np.sum(R ** 2 + I_prev * I_half) * dx)
        if track_energy:
            energy[0] = _expectation_energy(psi_now, V0, dx, self.m)

        save_pos = 0
        if 0 in save_set:
            psi_save[save_pos] = psi_now
            t_save[save_pos] = 0.0
            save_pos += 1

        # main loop ---------------------------------------------------------
        for n in range(n_steps):
            t_n = n * dt
            t_half = (n + 0.5) * dt
            V_half = self.potential(self.grid.x, t_half)

            # R^{n+1} = R^n + dt/hbar * H I^{n+1/2}     (using V at t_half)
            H_I = _stencil_apply_H(I_half, V_half, dx, self.m, self.bc)
            R = R + dt / units.HBAR * H_I

            # I^{n+3/2} = I^{n+1/2} - dt/hbar * H R^{n+1}    (V at t_{n+1})
            t_np1 = (n + 1) * dt
            V_np1 = self.potential(self.grid.x, t_np1)
            H_R = _stencil_apply_H(R, V_np1, dx, self.m, self.bc)
            I_new_half = I_half - dt / units.HBAR * H_R

            # Apply absorbing damp (if any) to both components.  Hard-wall
            # and periodic BCs need no mask -- the H stencil enforces them
            # and a multiplicative projection would break conservation.
            if damp is not None:
                R = R * damp
                I_new_half = I_new_half * damp

            # Diagnostics at integer step n+1.
            # I at n+1 reconstructed by averaging neighbours:
            #     I^{n+1} ~ 0.5*(I^{n+1/2} + I^{n+3/2})
            I_int = 0.5 * (I_half + I_new_half)
            psi_int = R + 1j * I_int
            norm[n + 1] = _norm_squared(psi_int, dx)
            norm_v[n + 1] = float(np.sum(R ** 2 + I_half * I_new_half) * dx)
            if track_energy:
                energy[n + 1] = _expectation_energy(psi_int, V_np1, dx, self.m)

            # Roll the half-step buffer.
            I_half = I_new_half

            if (n + 1) in save_set:
                psi_save[save_pos] = psi_int
                t_save[save_pos] = (n + 1) * dt
                save_pos += 1

        return SolverResult(grid=self.grid, times=t_save, psi=psi_save,
                            diag_times=diag_t, norm=norm,
                            norm_visscher=norm_v, energy=energy,
                            extras={"solver": self.name})


# ============================================================================
# 3. Crank-Nicolson (implicit, unitary, the gold standard)
# ============================================================================

class CrankNicolsonSolver(_BaseSolver):
    """Implicit Crank-Nicolson with sparse banded LU factorisation.

    Update rule:

    .. math::
        \\Big(1 + \\tfrac{i \\Delta t}{2 \\hbar} H\\Big)\\,\\psi^{n+1}
        = \\Big(1 - \\tfrac{i \\Delta t}{2 \\hbar} H\\Big)\\,\\psi^n.

    For a time-independent H we factorise the LHS once and re-use it.
    For a time-dependent H we re-factorise each step (still O(N) thanks
    to the tridiagonal structure).
    """

    name = "crank-nicolson"

    def _build_H(self, V: np.ndarray) -> sparse.csr_matrix:
        D2 = build_laplacian(self.grid, self.bc)
        T_op = -(units.HBAR ** 2) / (2.0 * self.m) * D2
        return T_op + sparse.diags(V, 0, format="csr")

    def run(self, psi0, n_steps, n_save=200, track_energy=True):
        N = self.grid.N
        dx = self.grid.dx
        dt = self.dt
        psi = psi0.astype(np.complex128).copy()

        is_time_dep = self.potential.is_time_dependent

        save_idx = self._save_indices(n_steps, n_save)
        save_set = set(int(i) for i in save_idx)
        psi_save = np.zeros((len(save_idx), N), dtype=np.complex128)
        t_save = np.zeros(len(save_idx))

        norm = np.zeros(n_steps + 1)
        energy = np.zeros(n_steps + 1) if track_energy else None
        diag_t = np.arange(n_steps + 1) * dt

        if self.bc == "absorbing" and self.boundary_mask is not None:
            damp = make_absorbing_step_mask(self.boundary_mask, dt)
        else:
            damp = None

        I = sparse.eye(N, format="csr", dtype=np.complex128)

        if not is_time_dep:
            V = self.potential(self.grid.x, 0.0)
            H = self._build_H(V)
            A = (I + 1j * dt / (2.0 * units.HBAR) * H).tocsc()
            B = (I - 1j * dt / (2.0 * units.HBAR) * H).tocsr()
            lu = splu(A)
        else:
            V = self.potential(self.grid.x, 0.0)

        save_pos = 0
        for n in range(n_steps + 1):
            if is_time_dep:
                V = self.potential(self.grid.x, n * dt)
            norm[n] = _norm_squared(psi, dx)
            if track_energy:
                energy[n] = _expectation_energy(psi, V, dx, self.m)
            if n in save_set:
                psi_save[save_pos] = psi
                t_save[save_pos] = n * dt
                save_pos += 1
            if n == n_steps:
                break

            if is_time_dep:
                V_half = self.potential(self.grid.x, (n + 0.5) * dt)
                H = self._build_H(V_half)
                A = (I + 1j * dt / (2.0 * units.HBAR) * H).tocsc()
                B = (I - 1j * dt / (2.0 * units.HBAR) * H).tocsr()
                rhs = B @ psi
                psi = sparse.linalg.spsolve(A, rhs)
            else:
                rhs = B @ psi
                psi = lu.solve(rhs)

            if damp is not None:
                psi = psi * damp

        return SolverResult(grid=self.grid, times=t_save, psi=psi_save,
                            diag_times=diag_t, norm=norm, energy=energy,
                            extras={"solver": self.name})


# ============================================================================
# 4. Split-operator (Fourier / Strang splitting)
# ============================================================================

class SplitOperatorSolver(_BaseSolver):
    r"""Fourier split-operator (Strang) solver.

    Strang splitting of the formal evolution operator
    :math:`U(\Delta t) = e^{-i H \Delta t/\hbar}` with
    :math:`H = T + V` gives the symmetric, second-order-accurate factorisation

    .. math::
        U(\Delta t) \;\approx\;
        e^{-i V \Delta t/(2\hbar)}\,
        e^{-i T \Delta t/\hbar}\,
        e^{-i V \Delta t/(2\hbar)},

    in which each factor is diagonal in *its own* basis: the kinetic
    propagator is diagonal in momentum space (eigenvalues
    :math:`\hbar^2 k^2/(2m)`) and the potential propagator is diagonal in
    position space.  Switching between the two representations with the
    discrete Fourier transform turns each step into

    .. math::
        \psi \to e^{-i V \Delta t/(2\hbar)} \psi
              \to \mathcal F^{-1}\!\Big[
                    e^{-i \hbar k^2 \Delta t/(2m)}\,
                    \mathcal F\psi\Big]
              \to e^{-i V \Delta t/(2\hbar)} \psi.

    Each individual factor is exactly unitary, so the scheme is unitary
    and unconditionally stable.  The local error is :math:`O(\Delta t^3)`
    per step / :math:`O(\Delta t^2)` global, the same order as leapfrog
    and Crank-Nicolson, but it has *no spatial discretisation error in
    the kinetic operator* (FFT differentiation is spectrally accurate)
    so it tends to converge with a much smaller pre-factor than the
    3-point stencil schemes.  Cost is :math:`O(N \log N)` per step.

    Natural boundary condition is *periodic* (the FFT assumes it).  For
    ``bc='hard'`` we additionally project psi to zero at the two endpoints
    after each step; for ``bc='absorbing'`` we apply the same exponential
    sponge as the other solvers.  Both compromise unitarity slightly --
    use ``bc='periodic'`` whenever possible for the cleanest comparison.
    """

    name = "split-operator"

    def run(self, psi0, n_steps, n_save=200, track_energy=True):
        N = self.grid.N
        dx = self.grid.dx
        dt = self.dt
        psi = psi0.astype(np.complex128).copy()

        is_time_dep = self.potential.is_time_dependent

        save_idx = self._save_indices(n_steps, n_save)
        save_set = set(int(i) for i in save_idx)
        psi_save = np.zeros((len(save_idx), N), dtype=np.complex128)
        t_save = np.zeros(len(save_idx))

        norm = np.zeros(n_steps + 1)
        energy = np.zeros(n_steps + 1) if track_energy else None
        diag_t = np.arange(n_steps + 1) * dt

        # Momentum-space wavenumbers used by the kinetic propagator.  numpy's
        # convention for the period is 2*pi / (N * dx), and fftfreq gives us
        # exactly that ordering (positive, then negative frequencies).
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        # Full kinetic propagator for one whole step.
        T_phase = np.exp(-1j * units.HBAR * k ** 2 * dt / (2.0 * self.m))

        # Hard / periodic / absorbing handling.
        # * periodic: do nothing -- FFT enforces it natively.
        # * hard: zero psi at the two endpoints after each step.
        # * absorbing: multiply by the exponential sponge mask each step.
        damp: Optional[np.ndarray] = None
        hard_endpoint_mask = False
        if self.bc == "absorbing" and self.boundary_mask is not None:
            damp = make_absorbing_step_mask(self.boundary_mask, dt)
        elif self.bc == "hard":
            hard_endpoint_mask = True

        # For a static V we can pre-compute the half-step potential propagator.
        if not is_time_dep:
            V_static = self.potential(self.grid.x, 0.0)
            V_half_phase = np.exp(-1j * V_static * dt / (2.0 * units.HBAR))
        else:
            V_static = None
            V_half_phase = None

        save_pos = 0
        for n in range(n_steps + 1):
            # Diagnostics use V at the current integer step time.
            V_now = (self.potential(self.grid.x, n * dt) if is_time_dep
                     else V_static)
            norm[n] = _norm_squared(psi, dx)
            if track_energy:
                energy[n] = _expectation_energy(psi, V_now, dx, self.m)
            if n in save_set:
                psi_save[save_pos] = psi
                t_save[save_pos] = n * dt
                save_pos += 1
            if n == n_steps:
                break

            # Strang step.  For time-dependent V the symmetric, second-order
            # variant uses V at the *integer* step (V_now) for the first
            # half-kick and V at (n+1)*dt for the second; for time-
            # independent V the two halves are identical.
            if is_time_dep:
                Vh1 = np.exp(-1j * V_now * dt / (2.0 * units.HBAR))
                psi = Vh1 * psi
            else:
                psi = V_half_phase * psi

            # Kinetic full-step in momentum space.
            psi = np.fft.ifft(T_phase * np.fft.fft(psi))

            if is_time_dep:
                V_next = self.potential(self.grid.x, (n + 1) * dt)
                Vh2 = np.exp(-1j * V_next * dt / (2.0 * units.HBAR))
                psi = Vh2 * psi
            else:
                psi = V_half_phase * psi

            if damp is not None:
                psi = psi * damp
            if hard_endpoint_mask:
                psi[0] = 0.0
                psi[-1] = 0.0

        return SolverResult(grid=self.grid, times=t_save, psi=psi_save,
                            diag_times=diag_t, norm=norm, energy=energy,
                            extras={"solver": self.name})


# ============================================================================
# Time-independent helpers: build H, find eigenstates, imaginary-time relax
# ============================================================================

def build_hamiltonian(grid: Grid1D, V: np.ndarray, bc: BCName = "hard",
                      m: float = units.M_E) -> sparse.csr_matrix:
    """Sparse ``(N, N)`` Hamiltonian ``H = -hbar^2/(2m) D2 + V`` for a
    static potential ``V``.  Matches the stencil used by the time-stepping
    solvers so eigenvalues of this H coincide with the spectrum the
    dynamics sees.
    """
    D2 = build_laplacian(grid, bc)
    T_op = -(units.HBAR ** 2) / (2.0 * m) * D2
    return (T_op + sparse.diags(np.asarray(V), 0, format="csr")).tocsr()


def bound_eigenstates(grid: Grid1D, V: np.ndarray, n_states: int,
                      bc: BCName = "hard", m: float = units.M_E):
    """Lowest ``n_states`` eigenpairs of the discretised Hamiltonian.

    Returns ``(E, phi)`` with ``E`` shape ``(n_states,)`` (in J, ascending)
    and ``phi`` shape ``(n_states, N)`` with each row :math:`L^2`-normalised
    on the grid (``int |phi_n|^2 dx = 1``).

    Implementation: sparse Lanczos (``scipy.sparse.linalg.eigsh`` with
    ``which='SA'``) on the sparse H.  For the grid sizes used in this
    project (N <= ~2000) this is milliseconds.
    """
    H = build_hamiltonian(grid, V, bc=bc, m=m)
    E, phi = eigsh(H, k=n_states, which="SA")
    order = np.argsort(E)
    E = E[order]
    phi = phi[:, order].T                      # (n_states, N)
    dx = grid.dx
    for n in range(n_states):
        phi[n] = phi[n] / np.sqrt(np.sum(np.abs(phi[n]) ** 2) * dx)
        # Fix a consistent global sign so repeated runs return the same
        # phase choice (purely cosmetic, useful for diff-ing snapshots).
        imax = int(np.argmax(np.abs(phi[n])))
        if phi[n, imax].real < 0:
            phi[n] = -phi[n]
    return E, phi


def ground_state_imag_time(
    grid: Grid1D,
    V: np.ndarray,
    d_tau: Optional[float] = None,
    n_steps: int = 2000,
    psi0: Optional[np.ndarray] = None,
    bc: BCName = "hard",
    m: float = units.M_E,
    tol: float = 1e-12,
):
    r"""Ground-state preparation by imaginary-time evolution.

    Replace :math:`t \to -i\tau` in the TDSE and it becomes a *diffusion*
    equation :math:`\partial_\tau \psi = -H\psi/\hbar` whose formal solution

    .. math::
        \psi(\tau) \propto \sum_n c_n e^{-E_n \tau/\hbar}\phi_n
                \;\xrightarrow{\tau \to \infty}\; c_0 \phi_0

    projects out the ground state exponentially fast, provided
    :math:`c_0 \neq 0`.  We implement the update with Crank-Nicolson in
    imaginary time,
    :math:`(1 + H\,d\tau/(2\hbar))\psi^{n+1} = (1 - H\,d\tau/(2\hbar))\psi^n`,
    followed by re-normalisation after each step.

    Returns
    -------
    psi_gs : (N,) ndarray (complex)
    E_gs   : float   (joules)
    energy_history : (n_iter,) ndarray -- <H> per iteration, for diagnostics
    """
    N = grid.N
    dx = grid.dx
    V_arr = np.asarray(V, dtype=float)

    if d_tau is None:
        # CN in imaginary time is unconditionally stable, so a larger d_tau
        # than the real-time CFL bound is fine.  We pick a value that makes
        # the per-step damping of E ~ 2 eV states already 1/e, so moderately
        # excited components die in ~20 iterations.
        T_scale = units.kinetic_energy_scale(dx, m)
        d_tau = 1.0 * units.HBAR / max(T_scale, 1e-60)

    if psi0 is None:
        x = grid.x
        x_mid = 0.5 * (grid.x_min + grid.x_max)
        sigma = 0.1 * grid.L
        psi0 = np.exp(-((x - x_mid) ** 2) / (2.0 * sigma ** 2))

    psi = psi0.astype(np.complex128).copy()
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

    H = build_hamiltonian(grid, V_arr, bc=bc, m=m)
    I_mat = sparse.eye(N, format="csr", dtype=np.complex128)
    A = (I_mat + 0.5 * d_tau / units.HBAR * H).tocsc()
    B = (I_mat - 0.5 * d_tau / units.HBAR * H).tocsr()
    lu = splu(A)

    E_hist = np.zeros(n_steps)
    E_prev: Optional[float] = None
    for k in range(n_steps):
        psi = lu.solve(B @ psi)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        E_hist[k] = float(np.real(np.conj(psi) @ (H @ psi)) * dx)
        if E_prev is not None:
            if abs(E_hist[k] - E_prev) / max(abs(E_prev), 1e-30) < tol:
                E_hist = E_hist[: k + 1]
                break
        E_prev = E_hist[k]

    return psi, float(E_hist[-1]), E_hist
