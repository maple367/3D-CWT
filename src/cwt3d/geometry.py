"""Single-cell dielectric geometry, lengths in um; array axes are (y, x).

Cells occupy [0, ax] x [0, ay]. Shapes must fit entirely in that cell;
coordinates are not implicitly wrapped. Callers may wrap periodic coordinates
explicitly. Permittivities are finite complex relative permittivities.
"""
from dataclasses import dataclass

import numpy as np


def _real(value):
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf':
        raise TypeError('Expected real numeric values')
    a = a.astype(float)
    if not np.all(np.isfinite(a)):
        raise ValueError('Values must be finite')
    return a


def _pair(value, name, positive=False):
    a = _real(value)
    if a.shape != (2,) or (positive and np.any(a <= 0)):
        raise ValueError(f'{name} must be a pair' + (' of positive values' if positive else ''))
    return tuple(a)


def _epsilon(value):
    a = np.asarray(value)
    if a.dtype.kind not in 'iufc':
        raise TypeError('Permittivity must be numeric')
    a = a.astype(complex)
    if not np.all(np.isfinite(a)):
        raise ValueError('Permittivity must be finite')
    return a


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return int(value)


def _materials(obj):
    for name in ('eps_background', 'eps_hole'):
        a = _epsilon(getattr(obj, name))
        if a.ndim:
            raise ValueError('Shape materials must be scalars')
        object.__setattr__(obj, name, complex(a))


def _cell(obj):
    p = _pair(obj.period_um, 'period_um', positive=True)
    c = tuple(v / 2 for v in p) if obj.center_um is None else _pair(obj.center_um, 'center_um')
    object.__setattr__(obj, 'period_um', p)
    object.__setattr__(obj, 'center_um', c)
    _materials(obj)


def _contained(lower, upper, period):
    if np.any(np.asarray(lower) < 0) or np.any(np.asarray(upper) > period):
        raise ValueError('Hole must fit entirely inside the cell; no clipping or wrapping')


@dataclass(frozen=True)
class CircleHole:
    """Circular inclusion; center defaults to the cell midpoint (all in um)."""
    radius_um: float
    period_um: tuple = (1., 1.)
    center_um: tuple | None = None
    eps_background: complex = 1.
    eps_hole: complex = 0.

    def __post_init__(self):
        _cell(self)
        r = _real(self.radius_um)
        if r.ndim or r <= 0:
            raise ValueError('radius_um must be a positive scalar')
        object.__setattr__(self, 'radius_um', float(r))
        _contained(np.array(self.center_um)-r, np.array(self.center_um)+r, self.period_um)

    @property
    def fill_fraction(self):
        return np.pi*self.radius_um**2 / np.prod(self.period_um)

    @property
    def mean_epsilon(self):
        return self.eps_background + (self.eps_hole-self.eps_background)*self.fill_fraction

    def __call__(self, x_um, y_um):
        x, y = _real(x_um), _real(y_um)
        inside = (x-self.center_um[0])**2 + (y-self.center_um[1])**2 <= self.radius_um**2
        return np.where(inside, self.eps_hole, self.eps_background)


@dataclass(frozen=True)
class RightTriangleHole:
    """Axis-aligned right triangle with independent +x/+y legs in um.

    center_um denotes the centroid, NOT the right-angle vertex. The default
    centroid is the cell midpoint. Equal legs give an isosceles right triangle;
    unequal legs are supported. Rotation is not part of this small API.
    """
    legs_um: tuple
    period_um: tuple = (1., 1.)
    center_um: tuple | None = None
    eps_background: complex = 1.
    eps_hole: complex = 0.

    def __post_init__(self):
        _cell(self)
        object.__setattr__(self, 'legs_um', _pair(self.legs_um, 'legs_um', positive=True))
        v = self.vertices_um
        _contained(v.min(axis=0), v.max(axis=0), self.period_um)

    @property
    def vertices_um(self):
        lx, ly = self.legs_um
        return np.array(self.center_um) + np.array([[-lx/3,-ly/3], [2*lx/3,-ly/3], [-lx/3,2*ly/3]])

    @property
    def fill_fraction(self):
        return np.prod(self.legs_um) / (2*np.prod(self.period_um))

    @property
    def mean_epsilon(self):
        return self.eps_background + (self.eps_hole-self.eps_background)*self.fill_fraction

    def __call__(self, x_um, y_um):
        x = (_real(x_um)-self.vertices_um[0,0])/self.legs_um[0]
        y = (_real(y_um)-self.vertices_um[0,1])/self.legs_um[1]
        return np.where((x >= 0) & (y >= 0) & (x+y <= 1), self.eps_hole, self.eps_background)


class SampledPermittivity:
    """Snapshot of cell-center POINT SAMPLES, not pixel-averaged epsilon.

    values[i,j] is epsilon at ((j+.5)*ax/Nx, (i+.5)*ay/Ny).
    Fourier transforms are discrete averages: continuous boundary/high-order
    coefficients require resolution convergence. No pixel sinc is implied.
    """
    def __init__(self, values, period_um=(1., 1.)):
        self._period_um = _pair(period_um, 'period_um', positive=True)
        a = _epsilon(values)
        if a.ndim != 2 or min(a.shape) == 0:
            raise ValueError('values must be a nonempty 2D array, axes (y,x)')
        # Keep only immutable storage/shape; never expose a shared ndarray's
        # mutable metadata, even indirectly through a public view's base chain.
        self._shape = a.shape
        self._data = a.tobytes()

    @property
    def period_um(self):
        return self._period_um

    @property
    def values(self):
        return np.ndarray(self._shape, dtype=complex, buffer=self._data)


def sample_permittivity(epsilon, period_um=(1.,1.), shape=(128,128)):
    """Evaluate a broadcast-compatible callable on a (Ny,Nx) midpoint grid."""
    p = _pair(period_um, 'period_um', positive=True)
    if len(shape) != 2:
        raise ValueError('shape must be (Ny,Nx)')
    ny, nx = (_integer(v, 'sample count') for v in shape)
    x = (np.arange(nx)+.5)*p[0]/nx
    y = (np.arange(ny)+.5)*p[1]/ny
    a = np.broadcast_to(_epsilon(epsilon(x[None,:], y[:,None])), (ny,nx))
    return SampledPermittivity(a, p)
