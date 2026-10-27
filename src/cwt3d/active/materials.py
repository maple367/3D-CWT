"""Local carrier material laws in um/ns units, with immutable parameter maps.

Densities are um^-3, gain um^-1, differential gain um^2, index coefficient
um^3, and A/B/C coefficients ns^-1, um^3/ns, um^6/ns. Signed trial carrier
values support implicit solvers; negative gain means absorption, not clipping.
"""
from dataclasses import dataclass

import numpy as np


def _real_array(value, name):
    """Validate before conversion: complex, bool and object are not accepted."""
    try:
        a = np.asarray(value)
        if a.dtype.kind not in 'iuf':
            raise ValueError(f'{name} must contain finite real values (not bool)')
        with np.errstate(over='raise', invalid='raise'):
            a = np.asarray(a, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise ValueError(f'{name} must contain finite real values')
        return a
    except (TypeError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f'{name} must contain representable finite real values') from exc


def _common_shape(*arrays):
    """Allow scalar expansion only, never singleton-axis/outer broadcasting."""
    shapes = {a.shape for a in arrays if a.ndim != 0}
    if len(shapes) > 1:
        raise ValueError('non-scalar inputs must have identical shapes')
    return next(iter(shapes), ())


def _finite_result(value):
    if not np.all(np.isfinite(value)):
        raise ValueError('material/carrier calculation produced nonfinite values')
    return value


def _parameter_property(index):
    def get(self):
        shape, data = self._parameters[index]
        if not shape:
            return float(np.frombuffer(data, dtype=np.float64)[0])
        return np.ndarray(shape, dtype=np.float64, buffer=data)
    return property(get)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class CarrierMaterial:
    """Immutable scalar/2-D parameters, with independent bytes-backed getters.

    Non-scalar maps must have one identical nonempty shape. Evaluation allows
    only scalar expansion; flat N vectors work with scalar parameters. N may
    be signed for Newton trials, U must be nonnegative. Overflow fails closed.
    """
    _parameters: tuple

    differential_gain_um2 = _parameter_property(0)
    transparency_density_um3 = _parameter_property(1)
    gain_compression_um3 = _parameter_property(2)
    index_per_density_um3 = _parameter_property(3)
    recombination_a_ns = _parameter_property(4)
    recombination_b_um3_ns = _parameter_property(5)
    recombination_c_um6_ns = _parameter_property(6)

    def __init__(self, differential_gain_um2, transparency_density_um3, *,
                 gain_compression_um3=0., index_per_density_um3=0.,
                 recombination_a_ns=0., recombination_b_um3_ns=0.,
                 recombination_c_um6_ns=0.):
        names = ('differential_gain_um2', 'transparency_density_um3',
                 'gain_compression_um3', 'index_per_density_um3',
                 'recombination_a_ns', 'recombination_b_um3_ns',
                 'recombination_c_um6_ns')
        values = (differential_gain_um2, transparency_density_um3,
                  gain_compression_um3, index_per_density_um3,
                  recombination_a_ns, recombination_b_um3_ns,
                  recombination_c_um6_ns)
        arrays = []
        for i, (name, value) in enumerate(zip(names, values)):
            a = _real_array(value, name)
            if a.ndim not in (0, 2) or a.size == 0:
                raise ValueError(f'{name} must be scalar or a nonempty 2-D map')
            if (i == 0 and np.any(a <= 0)) or (i not in (0, 3) and np.any(a < 0)):
                raise ValueError(f'{name} has an unphysical sign')
            arrays.append(a)
        _common_shape(*arrays)
        object.__setattr__(self, '_parameters', tuple((a.shape, a.tobytes()) for a in arrays))

    def _inputs(self, N, U=None):
        n = _real_array(N, 'N')
        if n.ndim > 2 or n.size == 0:
            raise ValueError('N must be a nonempty scalar, vector or 2-D map')
        params = tuple(np.ndarray(shape, dtype=np.float64, buffer=data)
                       for shape, data in self._parameters)
        if U is None:
            _common_shape(n, *params)
            return n, params
        u = _real_array(U, 'U')
        if np.any(u < 0) or u.ndim > 2 or u.size == 0:
            raise ValueError('U must be nonnegative scalar, vector or 2-D map')
        _common_shape(n, u, *params)
        return n, u, params

    def gain(self, N, U):
        """Signed gdiff*(N-Ntr)/(1+epsilon*U), in um^-1."""
        n, u, (g, nt, eps, *_) = self._inputs(N, U)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                return _finite_result(g*(n-nt)/(1.+eps*u))
        except FloatingPointError as exc:
            raise ValueError('gain calculation overflow or invalid arithmetic') from exc

    def index_change(self, N):
        """Signed dimensionless index shift relative to transparency."""
        n, (_, nt, _, coef, *_) = self._inputs(N)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                return _finite_result(coef*(n-nt))
        except FloatingPointError as exc:
            raise ValueError('index calculation overflow or invalid arithmetic') from exc

    def recombination(self, N):
        """A*N+B*N^2+C*N^3 in um^-3/ns, extended to signed trial N."""
        n, (_, _, _, _, a, b, c) = self._inputs(N)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                return _finite_result(((c*n+b)*n+a)*n)
        except FloatingPointError as exc:
            raise ValueError('recombination calculation overflow or invalid arithmetic') from exc
