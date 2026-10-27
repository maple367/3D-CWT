"""Sparse implicit active field/carrier evolution with the staggered mass B.

B is factored once; no inverse or dense Jacobian is constructed. The Jacobian
pattern includes the nonlocal triangular support of B^-1, not merely that of
A. Pattern construction costs O(ncell*max(Nx,Ny)); large patterns are rejected
explicitly rather than replaced with a dense approximation. No spontaneous
emission/noise, field renormalization or carrier clipping is performed.
"""
import numbers

import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse
from scipy.sparse.linalg import splu

from .results import ActiveState, ActiveResult, _numeric


_PATTERN_MAX_NNZ = 50_000_000


def _support(matrix):
    out = matrix.tocsr(copy=True)
    out.eliminate_zeros()
    return out.astype(bool)


def _jacobian_sparsity(grid, A, B, diffusion_matrix):
    """Conservative real (9*ncell)^2 sparsity, including mass-inverse fill.

    Component ordering is Rx, Sx, Ry, Sy. Inverse averaging is upstream
    triangular: Rx x<=current, Sx x>=current, Ry y<=current, Sy y>=current.
    Boolean multiplication avoids numeric path-count overflow/cancellation.
    """
    nx, ny = int(grid.Nx), int(grid.Ny)
    n = nx*ny
    rnnz = n*(nx+ny+2)
    I = sparse.eye(n,format='csr',dtype=bool)
    T = sparse.vstack([I]*4,format='csr')
    # Saturated gain depends on U=sum_components|Bv|²: even diagonal A
    # therefore couples every component in a cell before applying B^-1.
    local = _support(A) + _support(B) + T @ (T.T @ _support(B))
    max_degree = int(np.max(np.diff(local.indptr), initial=0))
    # Upper bound before allocating inverse support or product intermediates.
    upper = 4*rnnz*max_degree + 4*rnnz + 4*B.nnz + n + diffusion_matrix.nnz
    if upper > _PATTERN_MAX_NNZ:
        raise ValueError('nonlocal Jacobian sparsity exceeds 50 million entry construction bound; reduce grid')
    rows, cols = [], []
    for component in range(4):
        for cell in range(n):
            x, y = cell % nx, cell // nx
            if component == 0:
                upstream = np.arange(y*nx, cell+1)
            elif component == 1:
                upstream = np.arange(cell, (y+1)*nx)
            elif component == 2:
                upstream = np.arange(x, cell+1, nx)
            else:
                upstream = np.arange(cell, n, nx)
            rows.append(np.full(upstream.size, component*n+cell, dtype=np.int64))
            cols.append(upstream+component*n)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    R = sparse.csr_matrix((np.ones(rows.size,dtype=bool),(rows,cols)),shape=(4*n,4*n))
    ff = R @ local
    fn = R @ T
    nf = T.T @ _support(B)
    nn = I + _support(diffusion_matrix)
    return sparse.bmat([[ff,ff,fn],[ff,ff,fn],[nf,nf,nn]],format='csr',dtype=bool)


def _positive(value, name, *, infinite=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value,numbers.Real):
        raise ValueError(f'{name} must be a positive real scalar')
    value = float(value)
    if value <= 0 or np.isnan(value) or (not infinite and not np.isfinite(value)):
        raise ValueError(f'{name} must be positive and finite')
    return value


def _cell_map(value, shape, name, *, nonnegative=True):
    a = _numeric(value,name)
    if a.ndim == 0:
        a = np.full(shape,float(a))
    elif a.shape != shape:
        raise ValueError(f'{name} must be scalar or shape {shape}')
    if nonnegative and np.any(a < 0):
        raise ValueError(f'{name} must be nonnegative')
    return a.copy()


def simulate(cavity, material, initial, *, times_ns, current_density_A_um2=0.,
             diffusion_um2_ns=0., rtol=1e-6, field_atol=1e-10,
             carrier_atol=1e-3, max_step_ns=np.inf, method='BDF',
             max_output_bytes=256*1024**2, reference_detuning_um_inverse=0.):
    """Integrate an explicit seed using BDF or Radau in real coordinates.

    ``B dv/dt = i*vg*(A-reference_detuning*B)*v + tile(m,4)*(B*v)``;
    group velocity is already supplied by cavity. Fields are in the chosen
    rotating frame; add ``vg*reference_detuning`` (ns^-1) to phase rates or
    angular frequencies when returning to the original frame. Observables
    U, photon number and carriers do not depend on this reference.
    Newton trial carriers can be negative. Returned
    carriers below ``-carrier_atol`` raise; smaller excursions remain visible.
    Zero field stays zero (no spontaneous startup model). ``times_ns[0]`` is
    the initial state's time. Output memory guard conservatively includes
    solve_ivp, conversion and immutable-snapshot copies, not solver LU memory.
    This is a trajectory solver, not a stable-CW certification algorithm.
    """
    from .carriers import diffusion_operator, carrier_rhs

    if (isinstance(reference_detuning_um_inverse, (bool, np.bool_))
            or not isinstance(reference_detuning_um_inverse, numbers.Real)
            or not np.isfinite(reference_detuning_um_inverse)):
        raise ValueError('reference_detuning_um_inverse must be a finite real scalar')
    reference_detuning_um_inverse = float(reference_detuning_um_inverse)
    if not isinstance(initial,ActiveState):
        raise TypeError('initial must be an ActiveState with an explicit field seed')
    if method not in ('BDF','Radau'):
        raise ValueError('method must be BDF or Radau')
    rtol = _positive(rtol,'rtol')
    field_atol = _positive(field_atol,'field_atol')
    carrier_atol = _positive(carrier_atol,'carrier_atol')
    max_step_ns = _positive(max_step_ns,'max_step_ns',infinite=True)
    if isinstance(max_output_bytes,(bool,np.bool_)) or not isinstance(max_output_bytes,numbers.Integral) or max_output_bytes <= 0:
        raise ValueError('max_output_bytes must be a positive integer')
    t = _numeric(times_ns,'times_ns')
    if t.ndim != 1 or t.size < 2 or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError('times_ns must be nonnegative strictly increasing, length >= 2')
    shape = (int(cavity.grid.Ny),int(cavity.grid.Nx))
    n = shape[0]*shape[1]; q = 4*n
    if int(cavity.grid.n_dof) != q:
        raise ValueError('cavity must have four field components per cell')
    # Peak sampled-array storage estimate: real solver output, final conversion,
    # constructor validation copies and immutable bytes, plus sample indices.
    estimate = int(t.size)*(5*(q*16+n*8)+32)
    if estimate > max_output_bytes:
        raise ValueError(f'output storage estimate {estimate} exceeds max_output_bytes={max_output_bytes}')
    v = initial.field_vector
    if v.shape != (q,):
        raise ValueError(f'initial field must have shape {(q,)}')
    N = _cell_map(initial.carrier_density_um3,shape,'initial carrier density')
    current = current_density_A_um2
    if not callable(current):
        current = _cell_map(current,shape,'current_density_A_um2')
    def injection(time):
        return _cell_map(current(time) if callable(current) else current,
                         shape,'current_density_A_um2')
    injection(float(t[0]))
    diffusion = diffusion_operator(cavity.grid,diffusion_um2_ns).tocsr(copy=True)
    A, B = cavity.A.tocsr(copy=True), cavity.B.tocsc(copy=True)
    if A.shape != (q,q) or B.shape != (q,q):
        raise ValueError('invalid cavity matrix shapes')
    pattern = _jacobian_sparsity(cavity.grid,A,B,diffusion)
    try:
        lu = splu(B)
    except RuntimeError as exc:
        raise ValueError('cavity mass matrix B must be nonsingular') from exc
    def rhs(time,y):
        field = y[:q]+1j*y[q:2*q]
        carriers = y[2*q:].reshape(shape)
        centered = B @ field
        U = cavity.confinement/cavity.active_thickness_um*np.sum(
            abs(centered.reshape(4,*shape))**2,axis=0)
        m = np.asarray(cavity.material_rate(carriers,U,material))
        if m.shape != (n,) or not np.all(np.isfinite(m)):
            raise ValueError('material_rate must return finite flat ncell rates')
        dv = lu.solve(1j*cavity.vg_um_ns*(A @ field-reference_detuning_um_inverse*centered)
                      +np.tile(m,4)*centered)
        dn = np.asarray(carrier_rhs(carriers,U,injection(time),material,
                        vg_um_ns=cavity.vg_um_ns,
                        active_thickness_um=cavity.active_thickness_um,
                        diffusion_matrix=diffusion))
        if dn.shape != shape:
            raise ValueError('carrier_rhs must preserve the cell shape')
        out = np.concatenate((dv.real,dv.imag,dn.ravel()))
        if not np.all(np.isfinite(out)):
            raise ValueError('nonfinite coupled field/carrier derivative')
        return out
    y0 = np.concatenate((v.real,v.imag,N.ravel()))
    atol = np.concatenate((np.full(2*q,field_atol),np.full(n,carrier_atol)))
    solution = solve_ivp(rhs,(float(t[0]),float(t[-1])),y0,t_eval=t,
                         method=method,rtol=rtol,atol=atol,max_step=max_step_ns,
                         jac_sparsity=pattern,vectorized=False)
    if not solution.success or solution.y.shape != (9*n,t.size) or not np.array_equal(solution.t,t):
        raise RuntimeError(f'active integration failed: {solution.message}')
    fields = solution.y[:q].T+1j*solution.y[q:2*q].T
    carriers = solution.y[2*q:].T.reshape(t.size,*shape)
    if not np.all(np.isfinite(solution.y)):
        raise RuntimeError('active integration returned nonfinite state')
    if np.any(carriers < -carrier_atol):
        raise RuntimeError('active integration returned carrier density below -carrier_atol; refine tolerances/time steps')
    photons = np.array([cavity.photon_number(field) for field in fields])
    return ActiveResult(t,fields,carriers,photons,metadata={
        'method':method,'rtol':rtol,'field_atol':field_atol,'carrier_atol':carrier_atol,
        'nfev':int(solution.nfev),'njev':int(solution.njev),'nlu':int(solution.nlu),
        'radiation_scope':cavity.radiation_scope,'spontaneous_emission':False,
        'carrier_clipping':False,'field_renormalization':False,
        'output_storage_estimate_bytes':estimate,
        'mass_matrix_retained':True,
        'reference_detuning_um_inverse':reference_detuning_um_inverse,
        'phase_convention':'field is in rotating frame; add vg*reference_detuning to angular frequency (ns^-1)'})
