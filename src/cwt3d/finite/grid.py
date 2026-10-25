"""Face-staggered four-wave layout, exact incoming elimination, no PDE assembly.

Rows are y, columns x; lengths and coordinates are in micrometres (um).
Component-major free vectors follow conventions.BASIC_WAVE_ORDER, with each
(Ny,Nx) free block flattened in C order. No dummy corners or penalties.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..conventions import BASIC_WAVE_ORDER


def _snapshot(array):
    """Fresh ndarray metadata with an immutable bytes base, never a shared view."""
    a = np.asarray(array)
    return np.ndarray(a.shape, dtype=a.dtype, buffer=a.tobytes())


def _complex_array(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'biufc':
        raise ValueError(f'{name} must contain finite numeric values')
    with np.errstate(over='ignore', invalid='ignore'):
        a = np.asarray(a, dtype=np.complex128)
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must contain finite numeric values')
    return a


@dataclass(frozen=True, slots=True, init=False, eq=False)
class WaveFields:
    """Immutable snapshots of complex Rx,Sx,Ry,Sy arrays in BASIC_WAVE_ORDER.

    Each input must be a finite 2-D numeric array. Grid methods additionally
    check the exact staggered shapes. Interpolated fields instead have four
    (Ny,Nx) arrays. Every getter returns independent, read-only ndarray metadata
    backed directly by bytes; editing a getter's shape/dtype/strides cannot
    alter this object, its inputs, a grid or any other returned result.
    """
    _arrays: tuple

    def __init__(self, Rx, Sx, Ry, Sy):
        entries = []
        for name, value in zip(BASIC_WAVE_ORDER, (Rx, Sx, Ry, Sy)):
            a = _complex_array(value, name)
            if a.ndim != 2:
                raise ValueError(f'{name} must be a two-dimensional array')
            entries.append((a.shape, a.tobytes()))
        object.__setattr__(self, '_arrays', tuple(entries))

    def _get(self, index):
        shape, data = self._arrays[index]
        return np.ndarray(shape, dtype=np.complex128, buffer=data)

    @property
    def Rx(self):
        return self._get(0)

    @property
    def Sx(self):
        return self._get(1)

    @property
    def Ry(self):
        return self._get(2)

    @property
    def Sy(self):
        return self._get(3)


@dataclass(frozen=True, slots=True)
class StaggeredGrid:
    """Rectangular 0<=x<=Lx_um, 0<=y<=Ly_um grid of Nx by Ny cells.

    Rx,Sx: x-faces/y-centers, shape (Ny,Nx+1).
    Ry,Sy: x-centers/y-faces, shape (Ny+1,Nx).
    Only Rx[:,0], Sx[:,-1], Ry[0,:], Sy[-1,:] are constrained to zero.
    Opposite outgoing faces remain free, including on a one-cell domain.

    There are exactly 4*Nx*Ny free values: Rx[:,1:], Sx[:,:-1], Ry[1:,:],
    Sy[:-1,:], in that component order (BASIC_WAVE_ORDER), each C-flattened.
    These locations support four cell-centered equations per cell later;
    this module does not assemble A/B or solve a finite-domain eigenproblem.
    """
    Nx: int
    Ny: int
    Lx_um: float
    Ly_um: float

    def __post_init__(self):
        for name in ('Nx', 'Ny'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f'{name} must be a positive integer cell count')
            object.__setattr__(self, name, int(value))
        for name in ('Lx_um', 'Ly_um'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f'{name} must be a finite positive real length in um')
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive real length in um')
            object.__setattr__(self, name, value)
        try:
            spacing = (self.hx_um, self.hy_um)
        except OverflowError as exc:
            raise ValueError('cell counts must give representable spacing') from exc
        if not all(np.isfinite(h) and h > 0 for h in spacing):
            raise ValueError('derived spacing must be finite and nonzero')

    @property
    def n_dof(self):
        """Number of free complex amplitudes, exactly 4*Nx*Ny."""
        return 4*self.Nx*self.Ny

    @property
    def hx_um(self):
        return self.Lx_um/self.Nx

    @property
    def hy_um(self):
        return self.Ly_um/self.Ny

    def component_shape(self, component):
        """Full-array shape (rows y, columns x) of a named basic wave."""
        if component not in BASIC_WAVE_ORDER:
            raise ValueError(f'unknown component {component!r}')
        return (self.Ny, self.Nx+1) if component.endswith('x') else (self.Ny+1, self.Nx)

    def coordinates(self, component=None):
        """Return read-only (X_um,Y_um) meshes at faces, or cell centers (None).

        No cached public ndarray headers are shared. End faces are exactly
        0 and L; centers are arithmetic midpoints of their cell edges.
        """
        if component is not None:
            self.component_shape(component)
        x = np.linspace(0., self.Lx_um, self.Nx+1)
        y = np.linspace(0., self.Ly_um, self.Ny+1)
        if component not in ('Rx', 'Sx'):
            x = x[:-1]/2 + x[1:]/2
        if component not in ('Ry', 'Sy'):
            y = y[:-1]/2 + y[1:]/2
        return tuple(_snapshot(a) for a in np.meshgrid(x, y, indexing='xy'))

    def free_index(self, component, row, col):
        """Map a full face index to a free-vector id; incoming faces return -1.

        Negative/out-of-range indices raise IndexError, never Python-wrap.
        Integers only (including NumPy integers, excluding bool).
        """
        shape = self.component_shape(component)
        for value, bound in zip((row, col), shape):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError('row and col must be integers, not bool')
            if not 0 <= value < bound:
                raise IndexError('full face index outside component shape')
        row, col = int(row), int(col)
        if component == 'Rx':
            if col == 0:
                return -1
            col -= 1
        elif component == 'Sx':
            if col == self.Nx:
                return -1
        elif component == 'Ry':
            if row == 0:
                return -1
            row -= 1
        elif row == self.Ny:
            return -1
        return BASIC_WAVE_ORDER.index(component)*self.Nx*self.Ny + row*self.Nx + col

    @staticmethod
    def _free_part(component, a):
        if component == 'Rx':
            return a[:, 1:]
        if component == 'Sx':
            return a[:, :-1]
        if component == 'Ry':
            return a[1:, :]
        return a[:-1, :]

    def _validated_fields(self, fields):
        if not isinstance(fields, WaveFields):
            raise TypeError('fields must be WaveFields')
        arrays = tuple(getattr(fields, name) for name in BASIC_WAVE_ORDER)
        for name, a in zip(BASIC_WAVE_ORDER, arrays):
            if a.shape != self.component_shape(name):
                raise ValueError(f'{name} must have shape {self.component_shape(name)}')
        return arrays

    def unpack(self, vector):
        """Snapshot a finite (n_dof,) vector into four full complex fields.

        Constrained incoming entries are restored as exact complex zeros.
        The input is never modified and later input edits cannot affect fields.
        """
        v = _complex_array(vector, 'vector')
        if v.shape != (self.n_dof,):
            raise ValueError(f'vector must have shape ({self.n_dof},)')
        fields = []
        for name, block in zip(BASIC_WAVE_ORDER, v.reshape(4, self.Ny, self.Nx)):
            a = np.zeros(self.component_shape(name), dtype=np.complex128)
            self._free_part(name, a)[:] = block
            fields.append(a)
        return WaveFields(*fields)

    def pack(self, fields):
        """Return an independent read-only complex free-vector snapshot.

        Reject ANY nonzero incoming amplitude (no tolerance or silent loss).
        Full field shapes must match this grid, including the opposite face.
        """
        arrays = self._validated_fields(fields)
        incoming = (arrays[0][:, 0], arrays[1][:, -1], arrays[2][0, :], arrays[3][-1, :])
        if any(np.any(edge != 0) for edge in incoming):
            raise ValueError('incoming boundary amplitudes must be exactly zero')
        return _snapshot(np.concatenate([self._free_part(n, a).ravel(order='C')
                                         for n, a in zip(BASIC_WAVE_ORDER, arrays)]))

    def interpolate(self, fields):
        """Average adjacent faces onto (Ny,Nx) cell centers, no PDE or BC change.

        Rx/Sx use (left+right)/2, Ry/Sy use (bottom+top)/2. Accept arbitrary
        finite full fields, not just boundary-compatible ones. Half each
        operand first to avoid overflowing the sum of large finite values.
        """
        arrays = self._validated_fields(fields)
        centered = [a[:, :-1]/2+a[:, 1:]/2 if n.endswith('x')
                    else a[:-1, :]/2+a[1:, :]/2
                    for n, a in zip(BASIC_WAVE_ORDER, arrays)]
        return WaveFields(*centered)
