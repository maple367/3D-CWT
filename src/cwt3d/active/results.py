"""Immutable byte-backed active-cavity snapshots (um, ns, um^-3).

Public array headers are independent even across the entire ``.base`` chain.
Initial carriers must be nonnegative; results preserve small negative numerical
excursions rather than clipping or silently renormalizing either state.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


def _numeric(value, name, *, complex_ok=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'biufc' or (not complex_ok and np.iscomplexobj(raw)):
        raise ValueError(f'{name} must be numeric and real' if not complex_ok else f'{name} must be numeric')
    out = np.array(raw, dtype=np.complex128 if complex_ok else np.float64, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError(f'{name} must be finite')
    return out


def _snapshot(a):
    return (a.tobytes(order='C'), a.shape, a.dtype.str)


def _array(snapshot):
    data, shape, dtype = snapshot
    return np.ndarray(shape, dtype=np.dtype(dtype), buffer=data)


@dataclass(frozen=True, slots=True)
class _FrozenArray:
    snapshot: tuple


def _freeze(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufc':
            raise TypeError('metadata arrays must be numeric')
        return _FrozenArray(_snapshot(value))
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError('metadata keys must be strings')
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return value
    raise TypeError('metadata must contain only immutable scalar, numeric array, or mapping/sequence values')


def _thaw(value):
    if isinstance(value, _FrozenArray):
        return _array(value.snapshot)
    if isinstance(value, Mapping):
        return MappingProxyType({k: _thaw(v) for k, v in value.items()})
    if isinstance(value, tuple):
        return tuple(_thaw(v) for v in value)
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ActiveState:
    """Explicit field seed and nonnegative scalar or 2D carrier density."""
    _field: tuple
    _carrier: tuple

    def __init__(self, field_vector, carrier_density_um3):
        v = _numeric(field_vector, 'field_vector', complex_ok=True)
        n = _numeric(carrier_density_um3, 'carrier_density_um3')
        if v.ndim != 1 or v.size == 0:
            raise ValueError('field_vector must be a nonempty 1D array')
        if n.ndim not in (0, 2) or n.size == 0 or np.any(n < 0):
            raise ValueError('initial carrier density must be nonnegative scalar or 2D array')
        object.__setattr__(self, '_field', _snapshot(v))
        object.__setattr__(self, '_carrier', _snapshot(n))

    @property
    def field_vector(self):
        return _array(self._field)

    @property
    def carrier_density_um3(self):
        return _array(self._carrier)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ActiveResult:
    """Sampled trajectory; no interpolation, stability or CW certification.

    ``power_budget`` is a read-only mapping of numeric sample traces.
    """
    _times: tuple
    _fields: tuple
    _carriers: tuple
    _photons: tuple
    _metadata: object
    _power: object

    def __init__(self, times_ns, field_vectors, carrier_density_um3, photon_number,
                 *, metadata=None, power_budget=None):
        t = _numeric(times_ns, 'times_ns')
        v = _numeric(field_vectors, 'field_vectors', complex_ok=True)
        n = _numeric(carrier_density_um3, 'carrier_density_um3')
        p = _numeric(photon_number, 'photon_number')
        if t.ndim != 1 or t.size < 2 or np.any(t < 0) or np.any(np.diff(t) <= 0):
            raise ValueError('times_ns must be nonnegative and strictly increasing, length >= 2')
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError('metadata must be a mapping')
        if (v.ndim != 2 or n.ndim != 3 or n.size == 0 or p.shape != t.shape
                or v.shape[0] != t.size or n.shape[0] != t.size
                or v.shape[1] != 4*n.shape[1]*n.shape[2]):
            raise ValueError('trajectory shapes must be (nt,4*ncell), (nt,Ny,Nx), (nt,)')
        if np.any(p < 0):
            raise ValueError('photon_number must be nonnegative')
        power = {}
        for key, value in ({} if power_budget is None else power_budget).items():
            a = _numeric(value, 'power_budget')
            if a.shape != t.shape:
                raise ValueError('power budget traces must have shape (nt,)')
            power[key] = a
        for key, val in (('_times',t),('_fields',v),('_carriers',n),('_photons',p)):
            object.__setattr__(self,key,_snapshot(val))
        object.__setattr__(self,'_metadata',_freeze({} if metadata is None else metadata))
        object.__setattr__(self,'_power',_freeze(power))

    @property
    def times_ns(self):
        return _array(self._times)

    @property
    def field_vectors(self):
        return _array(self._fields)

    @property
    def carrier_density_um3(self):
        return _array(self._carriers)

    @property
    def photon_number(self):
        return _array(self._photons)

    @property
    def metadata(self):
        return _thaw(self._metadata)

    @property
    def power_budget(self):
        return _thaw(self._power)
