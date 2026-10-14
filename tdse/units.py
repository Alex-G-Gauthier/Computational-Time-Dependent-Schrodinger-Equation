"""SI constants and display-scale conversions.

We keep all internal arithmetic in SI (metres, seconds, joules, kilograms),
but values are typically input and plotted in the more human-friendly
*atomic-condensed-matter* units of nanometres, femtoseconds, and
electron-volts, with mass measured in electron masses.

Working in raw SI is fine numerically -- ``hbar ~ 1e-34 J s`` does not
under-flow IEEE 754 double precision -- as long as no quantity is squared
to extremes.  Choosing nm/fs/eV/m_e for the *display* layer keeps figures
legible and makes the dimensionless products in stability analysis (e.g.
``hbar*dt/(m*dx**2)``) numbers of order unity.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Fundamental SI constants (CODATA 2018) ---------------------------------
HBAR = 1.054_571_817e-34          # J s
H = 6.626_070_15e-34              # J s
M_E = 9.109_383_701_5e-31         # kg
E_CHARGE = 1.602_176_634e-19      # C
EV = E_CHARGE                     # 1 eV in joules
NM = 1.0e-9                       # m
FS = 1.0e-15                      # s
ANGSTROM = 1.0e-10                # m
A0 = 5.291_772_109_03e-11         # Bohr radius (m)
HARTREE = 4.359_744_722_2071e-18  # Hartree energy (J)


@dataclass(frozen=True)
class DisplayUnits:
    """Bundle of factors that map SI -> display units."""

    length: float = NM      # 1 display length unit, in metres
    length_label: str = "nm"
    time: float = FS        # in seconds
    time_label: str = "fs"
    energy: float = EV      # in joules
    energy_label: str = "eV"
    mass: float = M_E       # in kilograms
    mass_label: str = r"$m_e$"

    # Convenience accessors --------------------------------------------------
    def x_to(self, x_si):
        """metres -> display length."""
        return x_si / self.length

    def x_from(self, x_disp):
        """display length -> metres."""
        return x_disp * self.length

    def t_to(self, t_si):
        return t_si / self.time

    def t_from(self, t_disp):
        return t_disp * self.time

    def E_to(self, E_si):
        return E_si / self.energy

    def E_from(self, E_disp):
        return E_disp * self.energy

    def m_from(self, m_disp):
        return m_disp * self.mass


DEFAULT_UNITS = DisplayUnits()


# --- Useful derived scales --------------------------------------------------

def kinetic_energy_scale(dx_si: float, m_si: float = M_E) -> float:
    """The kinetic energy scale ``hbar^2/(2 m dx^2)`` set by grid spacing.

    This is the energy that *one grid spacing* of de-Broglie wavelength
    corresponds to; it is the natural energy unit for the discretised
    Hamiltonian and appears directly in the stability bound.
    """
    return HBAR ** 2 / (2.0 * m_si * dx_si ** 2)


def de_broglie_wavelength(E_si: float, m_si: float = M_E) -> float:
    """de-Broglie wavelength of a free particle of energy E."""
    import numpy as np

    p = (2.0 * m_si * E_si) ** 0.5
    return 2.0 * np.pi * HBAR / p


def momentum_from_energy(E_si: float, m_si: float = M_E) -> float:
    return (2.0 * m_si * E_si) ** 0.5


def wavenumber_from_energy(E_si: float, m_si: float = M_E) -> float:
    return momentum_from_energy(E_si, m_si) / HBAR


def period_of_state(E_si: float) -> float:
    """``T = 2*pi*hbar/E`` -- period of the phase rotation of an eigenstate."""
    import numpy as np

    return 2.0 * np.pi * HBAR / E_si
