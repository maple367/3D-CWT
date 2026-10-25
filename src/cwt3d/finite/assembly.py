"""Direct sparse finite-domain pencil for Liang thesis (4.21).

With lambda = delta + i alpha, A v = lambda B v and
A = C_lift B + i D. B averages adjacent faces onto cell centers; D is
(+d_x Rx, -d_x Sx, +d_y Ry, -d_y Sy). Incoming zero faces are omitted,
not penalized. Only uniform 4-by-4 coupling is supported. No eigensolver,
dense domain matrix, Kronecker matrix product or inverse is constructed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from ..conventions import BASIC_WAVE_ORDER
from .grid import StaggeredGrid, _complex_array


def _coupling_matrix(coupling):
    """Accept numeric (4,4), or the real CouplingResult.C in um^-1."""
    if hasattr(coupling, 'C'):
        # Lazy: assembly with a numeric matrix needs no optical-mode modules.
        from ..coupling import CouplingResult
        if not isinstance(coupling, CouplingResult):
            raise TypeError('coupling must be a 4x4 matrix or CouplingResult')
        if coupling.units != 'um^-1':
            raise ValueError('CouplingResult must use um^-1 units')
        coupling = coupling.C
    c = _complex_array(coupling, 'coupling')
    if c.shape != (4, 4):
        raise ValueError('uniform coupling must have shape (4, 4)')
    return c.copy()


def _check_grid(grid):
    if not isinstance(grid, StaggeredGrid):
        raise TypeError('grid must be a StaggeredGrid')
    # Python ints first: do not overflow np.int32/int64 during dimension math.
    # NumPy allocation lengths/byte sizes and sparse indices use signed intp.
    limit = int(np.iinfo(np.intp).max)
    n = grid.Nx*grid.Ny
    if (grid.Nx+1 > limit or grid.Ny+1 > limit or grid.n_dof+1 > limit
            or 32*n > limit//np.dtype(np.complex128).itemsize):
        raise ValueError('grid dimensions/COO storage exceed representable index range')
    with np.errstate(over='ignore', divide='ignore'):
        inverse_h = (1/grid.hx_um, 1/grid.hy_um)
    if not all(np.isfinite(h) for h in inverse_h):
        raise ValueError('inverse grid spacing must be finite and representable')
    return n, inverse_h


def _csr(rows, cols, values, size):
    """Convert only one-dimensional triplets, merging any duplicate entries."""
    with np.errstate(over='ignore', invalid='ignore'):
        matrix = coo_matrix((np.concatenate(values),
                             (np.concatenate(rows), np.concatenate(cols))),
                            shape=(size, size), dtype=np.complex128).tocsr()
        matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError('assembled coefficients exceed finite numeric range')
    return matrix


def _assemble(grid, c, n, inverse_h):
    """Vectorized local face templates, O(N) work and memory (N=Nx*Ny)."""
    cell = np.arange(n, dtype=np.intp)
    x, y = cell % grid.Nx, cell // grid.Nx
    # Each tuple contains low/high face free ids. A -1 means eliminated BC.
    faces = (
        (np.where(x == 0, -1, cell-1), cell),
        (cell+n, np.where(x == grid.Nx-1, -1, cell+n+1)),
        (np.where(y == 0, -1, cell+2*n-grid.Nx), cell+2*n),
        (cell+3*n, np.where(y == grid.Ny-1, -1, cell+3*n+grid.Nx)),
    )
    ar, ac, av, br, bc, bv, dr, dc, dv = ([] for _ in range(9))
    for source, pair in enumerate(faces):
        sign_over_h = (1 if source % 2 == 0 else -1)*inverse_h[source//2]
        for side, indices in enumerate(pair):
            mask = indices >= 0
            local = cell[mask]
            cols = indices[mask]
            rows = source*n+local
            half = np.full(cols.size, .5, dtype=np.complex128)
            difference = np.full(cols.size, (-1 if side == 0 else 1)*sign_over_h,
                                 dtype=np.complex128)
            br.append(rows)
            bc.append(cols)
            bv.append(half)
            dr.append(rows)
            dc.append(cols)
            dv.append(difference)
            for receiver in range(4):
                # Fuse coupling and same-component transport before COO.
                value = np.full(cols.size, c[receiver, source]/2, dtype=np.complex128)
                if receiver == source:
                    with np.errstate(over='ignore', invalid='ignore'):
                        value += 1j*difference
                ar.append(receiver*n+local)
                ac.append(cols)
                av.append(value)
    size = 4*n
    return (_csr(ar, ac, av, size), _csr(br, bc, bv, size),
            _csr(dr, dc, dv, size))


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FinitePencil:
    """Immutable assembly owner with independent public CSR snapshots.

    grid is immutable. A and D have units um^-1; B is dimensionless.
    Both rows (cell centers) and columns (free faces) are component-major
    (Rx,Sx,Ry,Sy), each block C-flattened (row=y, column=x).
    B is generally non-Hermitian, NOT an SPD mass matrix; do not feed it
    blindly to a Hermitian/generalized eigensolver's mass argument.

    Every A/B/D access copies CSR data, indices, indptr and shape: ordinary
    public mutation cannot change this pencil or other reads. This costs
    O(nnz) per getter; a future solver should fetch A and B just once.
    C returns independent ndarray metadata over immutable bytes. There is
    no public accessor exposing a shared internal sparse object.
    """
    grid: StaggeredGrid
    _A: object
    _B: object
    _D: object
    _c_bytes: bytes

    def __init__(self, grid, coupling):
        n, inverse_h = _check_grid(grid)
        c = _coupling_matrix(coupling)
        a, b, d = _assemble(grid, c, n, inverse_h)
        object.__setattr__(self, 'grid', grid)
        object.__setattr__(self, '_A', a)
        object.__setattr__(self, '_B', b)
        object.__setattr__(self, '_D', d)
        object.__setattr__(self, '_c_bytes', c.tobytes())

    @property
    def A(self):
        return self._A.copy()

    @property
    def B(self):
        return self._B.copy()

    @property
    def D(self):
        """Signed face-difference operator, before multiplication by i."""
        return self._D.copy()

    @property
    def C(self):
        return np.ndarray((4, 4), dtype=np.complex128, buffer=self._c_bytes)

    @property
    def ordering(self):
        return BASIC_WAVE_ORDER

    @property
    def units(self):
        return 'um^-1'

    @property
    def B_units(self):
        return '1'

    @property
    def eigenvalue_convention(self):
        return 'lambda = delta + i alpha'


def assemble_pencil(grid, coupling):
    """Assemble A v=lambda B v directly; coupling is (4,4) or CouplingResult.

    Lengths use um, coupling uses um^-1. Only Rx-left, Sx-right, Ry-bottom
    and Sy-top are zero. There are exactly 4*Nx*Ny unknowns and equations.
    This function does not solve or invert either matrix.
    """
    return FinitePencil(grid, coupling)
