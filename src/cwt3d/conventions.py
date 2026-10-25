"""Shared mathematical conventions for the square-lattice TE reference core.

Fields use exp(+i*omega*t) and exp(-i*beta0*(m*x+n*y)). Fourier
coefficients use the positive exponent and the unit-cell average (1/area).
The four TE amplitudes are (Rx, Sx, Ry, Sy): the x-travelling waves
contribute to Ey and the y-travelling waves to Ex, with no extra sign.
Thus Ey = Rx*exp(-i*beta0*x) + Sx*exp(+i*beta0*x), and
Ex = Ry*exp(-i*beta0*y) + Sy*exp(+i*beta0*y), before any common
longitudinal profile. Its normalization belongs to the future waveguide API.

Internal lengths are micrometres (um); wavenumbers and amplitude losses
are um^-1, never implicitly cm^-1. E(s)=E(0)*exp(-alpha*s) defines
positive amplitude loss alpha; power decays as exp(-2*alpha*s).
Negative alpha denotes gain. No eigenvalue-to-alpha rule is imposed here.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

TIME_EXPONENT_SIGN = 1
SPATIAL_EXPONENT_SIGN = -1
FOURIER_COEFFICIENT_EXPONENT_SIGN = 1
INTERNAL_LENGTH_UNIT = "um"
INTERNAL_INVERSE_LENGTH_UNIT = "um^-1"


@dataclass(frozen=True)
class BasicWave:
    """Immutable harmonic index and transverse electric-field component."""

    name: str
    m: int
    n: int
    electric_component: Literal["Ex", "Ey"]


BASIC_WAVES = (
    BasicWave("Rx", 1, 0, "Ey"),
    BasicWave("Sx", -1, 0, "Ey"),
    BasicWave("Ry", 0, 1, "Ex"),
    BasicWave("Sy", 0, -1, "Ex"),
)
BASIC_WAVE_ORDER = tuple(wave.name for wave in BASIC_WAVES)


def time_phase(omega_rad_per_time: ArrayLike, t: ArrayLike):
    """Return exp(+i*omega*t); omega and t must use matching time units."""
    return np.exp(1j * TIME_EXPONENT_SIGN * np.asarray(omega_rad_per_time) * np.asarray(t))


def spatial_phase(m: int, n: int, *, beta0_per_um: ArrayLike,
                  x_um: ArrayLike, y_um: ArrayLike):
    """Return exp(-i*beta0*(m*x+n*y)) with NumPy broadcasting.

    Coordinates must be in um and beta0 in um^-1; m,n are the signed
    integer harmonic indices. This is a basis phase, not a Fourier solver.
    """
    argument = np.asarray(beta0_per_um) * (m * np.asarray(x_um) + n * np.asarray(y_um))
    return np.exp(1j * SPATIAL_EXPONENT_SIGN * argument)


def fourier_coefficient_phase(m: int, n: int, *, beta0_per_um: ArrayLike,
                              x_um: ArrayLike, y_um: ArrayLike):
    """Return the positive-exponent weight for the (m,n) coefficient.

    Multiply the field by this weight and average over one unit cell.
    Coordinates and beta0 use um and um^-1; no integration is done here.
    """
    argument = np.asarray(beta0_per_um) * (m * np.asarray(x_um) + n * np.asarray(y_um))
    return np.exp(1j * FOURIER_COEFFICIENT_EXPONENT_SIGN * argument)


def _real_finite(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError("Expected real numeric values, not boolean, complex or text")
    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Values must be finite")
    return array


def _scaled(value: ArrayLike, scale: float) -> float | NDArray[np.float64]:
    with np.errstate(over="raise", invalid="raise"):
        result = _real_finite(value) * scale
    return float(result) if result.ndim == 0 else result


def convert_length(value: ArrayLike, *, from_unit: Literal["um", "cm"],
                   to_unit: Literal["um", "cm"]) -> float | NDArray[np.float64]:
    """Convert real finite lengths between explicit 'um' and 'cm'.

    Both units are required keywords; inverse-length/unknown units fail
    rather than being guessed. Scalars return float; arrays return float64
    arrays. Signed coordinates are allowed. Input arrays are not modified.
    """
    scales = {"um": 1.0, "cm": 1e4}  # length in internal um
    if from_unit not in scales or to_unit not in scales:
        raise ValueError("Length units must be 'um' or 'cm'")
    return _scaled(value, scales[from_unit] / scales[to_unit])


def convert_inverse_length(value: ArrayLike, *,
                           from_unit: Literal["um^-1", "cm^-1"],
                           to_unit: Literal["um^-1", "cm^-1"]) -> float | NDArray[np.float64]:
    """Convert real finite wavenumbers/losses, preserving their sign.

    Explicit unit strings prevent confusing lengths with inverse lengths:
    1 um^-1 = 1e4 cm^-1. Scalars return float; arrays return float64
    arrays. This does not convert amplitude loss into power loss.
    """
    scales = {"um^-1": 1.0, "cm^-1": 1e-4}
    if from_unit not in scales or to_unit not in scales:
        raise ValueError("Inverse-length units must be 'um^-1' or 'cm^-1'")
    return _scaled(value, scales[from_unit] / scales[to_unit])


def amplitude_loss_to_power_loss(alpha_per_um: ArrayLike) -> float | NDArray[np.float64]:
    """Return power loss 2*alpha in um^-1 from amplitude loss in um^-1.

    E(s)=E(0)*exp(-alpha*s), so |E(s)|^2=|E(0)|^2*exp(-2*alpha*s).
    Signed real finite scalar/array inputs are supported; negative is gain.
    Convert cm^-1 explicitly with convert_inverse_length before calling.
    """
    return _scaled(alpha_per_um, 2.0)
