"""Conservative cell-centred carrier transport and local balance (um, ns)."""
import numpy as np
from scipy import sparse
from scipy.constants import elementary_charge

from ..finite.grid import StaggeredGrid
from .materials import CarrierMaterial, _real_array, _finite_result


def _scalar(value, name, *, positive=False):
    a = _real_array(value, name)
    if a.ndim != 0 or (a <= 0 if positive else a < 0):
        raise ValueError(f'{name} must be a finite {"positive" if positive else "nonnegative"} scalar')
    return float(a)


def diffusion_operator(grid, diffusion_um2_ns):
    """Real CSR finite-volume D*Laplacian, no-flux outer faces, C-order cells.

    D is a constant nonnegative scalar in um^2/ns. Every internal face adds
    equal and opposite fluxes; one-cell axes have no faces and no diffusion.
    """
    if not isinstance(grid, StaggeredGrid):
        raise ValueError('grid must be a StaggeredGrid')
    d = _scalar(diffusion_um2_ns, 'diffusion_um2_ns')
    nx, ny = grid.Nx, grid.Ny
    size = nx*ny
    if d == 0:
        return sparse.csr_matrix((size, size), dtype=float)
    ids = np.arange(size).reshape(ny, nx)
    rows, cols, values = [], [], []
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for left, right, h in ((ids[:, :-1], ids[:, 1:], grid.hx_um),
                                   (ids[:-1, :], ids[1:, :], grid.hy_um)):
                if left.size == 0:
                    continue
                rate = np.float64(d)/h/h
                _finite_result(rate)
                i, j = left.ravel(), right.ravel()
                rows.extend((i, j, i, j))
                cols.extend((j, i, i, j))
                values.extend((np.full(i.size, rate), np.full(i.size, rate),
                               np.full(i.size, -rate), np.full(i.size, -rate)))
            if not rows:
                return sparse.csr_matrix((size, size), dtype=float)
            result = sparse.coo_matrix((np.concatenate(values),
                                        (np.concatenate(rows), np.concatenate(cols))),
                                       shape=(size, size)).tocsr()
            _finite_result(result.data)
            result.eliminate_zeros()
            return result
    except FloatingPointError as exc:
        raise ValueError('diffusion coefficient/spacing caused overflow') from exc


def _control(value, name, shape):
    a = _real_array(value, name)
    if a.ndim != 0 and a.shape != shape:
        raise ValueError(f'{name} must be scalar or match N shape {shape}')
    if np.any(a < 0):
        raise ValueError(f'{name} must be nonnegative')
    return a


def carrier_rhs(N, U, current_density_A_um2, material, *, vg_um_ns,
                active_thickness_um, diffusion_matrix):
    """Return dN/dt (um^-3/ns), preserving exactly the input N shape.

    N may be scalar, flat, or cell-shaped; flat N requires scalar material
    parameters. U and J are scalar or exactly N-shaped, never outer-broadcast.
    J is A/um^2; J*1e-9/(e*d) converts charge/second to carriers/ns/volume.
    Signed trial N and gain are preserved. The matrix must be finite real
    sparse and N.size-square (typically from diffusion_operator).
    """
    if not isinstance(material, CarrierMaterial):
        raise ValueError('material must be a CarrierMaterial')
    n = _real_array(N, 'N')
    if n.ndim > 2 or n.size == 0:
        raise ValueError('N must be a nonempty scalar, vector or 2-D map')
    u = _control(U, 'U', n.shape)
    j = _control(current_density_A_um2, 'current_density_A_um2', n.shape)
    vg = _scalar(vg_um_ns, 'vg_um_ns', positive=True)
    thickness = _scalar(active_thickness_um, 'active_thickness_um', positive=True)
    if (not sparse.issparse(diffusion_matrix) or
            diffusion_matrix.shape != (n.size, n.size) or
            diffusion_matrix.dtype.kind not in 'iuf'):
        raise ValueError('diffusion_matrix must be real sparse and N.size-square')
    matrix = diffusion_matrix.tocsr()
    _real_array(matrix.data, 'diffusion_matrix')
    gain = material.gain(n, u)
    recombination = material.recombination(n)
    if np.shape(gain) != n.shape or np.shape(recombination) != n.shape:
        raise ValueError('material evaluation must match N shape exactly')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            # Divide sequentially to avoid underflow in e*thickness.
            injection = j*1e-9/elementary_charge/thickness
            transport = (matrix @ n.ravel(order='C')).reshape(n.shape)
            result = injection-recombination-vg*gain*u+transport
            return np.asarray(_finite_result(result)).reshape(n.shape)
    except FloatingPointError as exc:
        raise ValueError('carrier balance overflow or invalid arithmetic') from exc
