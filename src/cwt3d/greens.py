"""Outgoing scalar Green kernel of an infinite homogeneous passive medium.

Lengths are um, k0 and k_parallel are um^-1, epsilon is relative permittivity.
With exp(+i*omega*t), passive media have Im(epsilon) <= 0. This solves

    [d_z^2 + k0^2*epsilon - k_parallel^2] G(z,z') = -delta(z-z').

Set gamma = sqrt(k_parallel^2-k0^2*epsilon), on the branch Re(gamma)>=0
and Im(gamma)>=0 (including the lossless propagating boundary). Then
G=exp(-gamma*abs(z-z'))/(2*gamma). Equivalently kz=-i*gamma has Re(kz)>=0,
Im(kz)<=0 and G=-i*exp(-i*kz*abs(z-z'))/(2*kz). The derivative jump at the
source is -1. No growing exponential or legacy exp(i*gamma*d) is used.

Only the infinite homogeneous scalar problem is implemented, not interfaces,
layer reflections, TM matching, harmonic enumeration or a coupling matrix.
"""
from __future__ import annotations

import cmath
import math

import numpy as np


class CutoffError(ValueError):
    """No finite unique outgoing kernel at numerically exact gamma=0."""


def _contains_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_bool(item) for item in value)
    return isinstance(value, np.ndarray) and value.dtype.kind == 'b'


def _numeric(value, name, *, complex_allowed=False):
    if _contains_bool(value):
        raise ValueError(f'{name} must be numeric, not boolean')
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite numeric value') from exc
    kinds = 'iufc' if complex_allowed else 'iuf'
    if array.dtype.kind not in kinds or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite and ' + ('numeric' if complex_allowed else 'real'))
    return array


def _scalar(value, name, *, complex_allowed=False):
    array = _numeric(value, name, complex_allowed=complex_allowed)
    if array.ndim != 0:
        raise ValueError(f'{name} must be scalar')
    result = complex(array) if complex_allowed else float(array)
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError(f'{name} is outside finite float64 range')
    return result


class HomogeneousGreen:
    """Read-only configuration for an infinite homogeneous Green kernel.

    Parameters: finite real k0>0, finite passive complex epsilon, and finite
    real k_parallel>=0 (default zero). All are scalar; to change configuration
    construct a new kernel. Zero-dimensional NumPy inputs are copied to Python
    scalars, so there is no mutable array or metadata alias.

    Coordinates are finite real scalars/arrays in um and follow NumPy
    broadcasting. For two independent grids use z[:,None], z_prime[None,:].
    Outputs are complex with the broadcast shape; evaluation has no cache.

    Exact float64 cutoff raises CutoffError, including when subtraction has
    rounded an unresolved near-cutoff case to zero. Nonzero gamma is never
    tolerance-clamped. Extreme parameters/intermediates outside float64 range
    are rejected, not claimed to have a resolved physical cutoff.
    """

    __slots__ = ('_k0', '_epsilon', '_k_parallel', '_gamma')

    def __init__(self, k0, epsilon, k_parallel=0):
        k0 = _scalar(k0, 'k0')
        epsilon = _scalar(epsilon, 'epsilon', complex_allowed=True)
        k_parallel = _scalar(k_parallel, 'k_parallel')
        if k0 <= 0 or k_parallel < 0:
            raise ValueError('k0 must be positive and k_parallel nonnegative')
        if epsilon.imag > 0:
            raise ValueError('active epsilon is unsupported: exp(+i*omega*t) requires Im(epsilon)<=0')
        k0_squared = k0 * k0
        parallel_squared = k_parallel * k_parallel
        if not math.isfinite(k0_squared) or not math.isfinite(parallel_squared) or k0_squared == 0:
            raise ValueError('squared wavenumber outside resolved float64 range')
        radicand = parallel_squared - k0_squared * epsilon
        if not (math.isfinite(radicand.real) and math.isfinite(radicand.imag)):
            raise ValueError('derived gamma squared outside finite float64 range')
        if radicand == 0:
            raise CutoffError('numerically exact cutoff: gamma squared is zero; exact or unresolved near-cutoff')
        # The passive radicand lies in the upper half plane. Force +0j on
        # its real axis: signed negative zero must not select the incoming root.
        gamma = cmath.sqrt(complex(radicand.real, max(0., radicand.imag)))
        self._k0 = k0
        self._epsilon = epsilon
        self._k_parallel = k_parallel
        self._gamma = gamma

    @property
    def k0(self):
        return self._k0

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def k_parallel(self):
        return self._k_parallel

    @property
    def gamma(self):
        """Decay exponent in um^-1, Re>=0 and Im>=0 for passive epsilon."""
        return self._gamma

    @property
    def kz(self):
        """Outgoing longitudinal wavenumber -i*gamma, in um^-1."""
        return -1j * self._gamma

    def _difference(self, z, z_prime):
        z = _numeric(z, 'z')
        z_prime = _numeric(z_prime, 'z_prime')
        # Convert integer inputs before subtraction to avoid integer wraparound.
        try:
            with np.errstate(over='raise', invalid='raise'):
                difference = np.asarray(z, dtype=float) - np.asarray(z_prime, dtype=float)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError('coordinate difference outside finite float64 range') from exc
        return difference

    def _exponential(self, difference):
        try:
            with np.errstate(over='raise', invalid='raise'):
                exponent = -self._gamma * np.abs(difference)
                return np.exp(exponent)
        except FloatingPointError as exc:
            raise ValueError('propagation exponent outside finite float64 range') from exc

    def __call__(self, z, z_prime):
        """Continuous G(z,z_prime), including its finite value at the source."""
        difference = self._difference(z, z_prime)
        # Divide in this order so forming 2*gamma need not overflow.
        return (0.5 / self._gamma) * self._exponential(difference)

    def derivative(self, z, z_prime, *, side=None):
        """Receiver derivative dG/dz; source limits are +1/2 left, -1/2 right.

        At z==z_prime, side='left' or 'right' is required. The default rejects
        an entire batch containing any source point rather than inventing a
        continuous derivative. side affects only exact source coincidences.
        """
        if side is not None and (not isinstance(side, str) or side not in ('left', 'right')):
            raise ValueError("side must be None, 'left' or 'right'")
        difference = self._difference(z, z_prime)
        at_source = difference == 0
        if side is None and np.any(at_source):
            raise ValueError("derivative at the source requires side='left' or 'right'")
        sign = np.sign(difference)
        if side is not None:
            sign = np.where(at_source, -1 if side == 'left' else 1, sign)
        return -0.5 * sign * self._exponential(difference)
