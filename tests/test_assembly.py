"""Task 9: independent matrix, field, flux and continuous-center oracles."""
from dataclasses import FrozenInstanceError
import inspect
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from cwt3d.finite import StaggeredGrid, WaveFields
from cwt3d.finite.assembly import FinitePencil, assemble_pencil

NAMES = ('Rx', 'Sx', 'Ry', 'Sy')


def complex_c():
    rng = np.random.default_rng(503)
    return rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))


def dense_oracle(nx, ny, lx, ly, c):
    """Enumerate full faces independently, never using grid.free_index."""
    maps, counter = [], 0
    for k in range(4):
        shape = (ny, nx+1) if k < 2 else (ny+1, nx)
        table = {}
        for y in range(shape[0]):
            for x in range(shape[1]):
                incoming = ((k == 0 and x == 0) or (k == 1 and x == nx)
                            or (k == 2 and y == 0) or (k == 3 and y == ny))
                if not incoming:
                    table[y, x] = counter
                    counter += 1
        maps.append(table)
    assert counter == 4*nx*ny
    b = np.zeros((counter, counter), complex)
    d = np.zeros_like(b)
    for k in range(4):
        h = lx/nx if k < 2 else ly/ny
        sign = 1 if k % 2 == 0 else -1
        for y in range(ny):
            for x in range(nx):
                row = k*nx*ny+y*nx+x
                faces = ((y, x), (y, x+1)) if k < 2 else ((y, x), (y+1, x))
                for pos, face in enumerate(faces):
                    if face in maps[k]:
                        col = maps[k][face]
                        b[row, col] += .5
                        d[row, col] += sign*(-1 if pos == 0 else 1)/h
    return np.kron(c, np.eye(nx*ny)) @ b + 1j*d, b, d


def field_oracle(g, v, c):
    f = g.unpack(v)
    arrays = [getattr(f, n) for n in NAMES]
    center = np.array([(a[:, :-1]+a[:, 1:])/2 if k < 2
                       else (a[:-1, :]+a[1:, :])/2 for k, a in enumerate(arrays)])
    diff = np.array([(a[:, 1:]-a[:, :-1])/g.hx_um if k < 2
                     else (a[1:, :]-a[:-1, :])/g.hy_um for k, a in enumerate(arrays)])
    diff[[1, 3]] *= -1
    return (c @ center.reshape(4, -1) + 1j*diff.reshape(4, -1)).ravel(), center.ravel(), diff.ravel()


def test_01_one_cell_exact_hand_matrix():
    c = np.array([[1+2j, 2, 3j, -4], [5, -6j, 7, 8],
                  [9j, 10, 11, 12j], [13, 14j, 15, -16]], complex)
    p = assemble_pencil(StaggeredGrid(1, 1, 2, 5), c)
    assert isinstance(p, FinitePencil)
    np.testing.assert_array_equal(p.B.toarray(), .5*np.eye(4))
    np.testing.assert_array_equal(p.D.toarray(), np.diag([.5, .5, .2, .2]))
    np.testing.assert_array_equal(p.A.toarray(), .5*c + 1j*np.diag([.5, .5, .2, .2]))
    assert p.A.dtype == p.B.dtype == np.dtype(complex)


@pytest.mark.parametrize('nx,ny,lx,ly', [(2, 3, 7., 11.), (1, 7, 2., 13.), (8, 1, 9., 3.), (4, 5, .7, 2.3)])
@pytest.mark.parametrize('zero', [False, True])
def test_02_independent_dense_and_actual_field_oracles(nx, ny, lx, ly, zero):
    c = np.zeros((4, 4)) if zero else complex_c()
    g = StaggeredGrid(nx, ny, lx, ly)
    p = assemble_pencil(g, c)
    expected = dense_oracle(nx, ny, lx, ly, c)
    for actual, wanted in zip((p.A, p.B, p.D), expected):
        assert sparse.isspmatrix_csr(actual)
        assert actual.has_canonical_format and actual.has_sorted_indices
        assert np.all(actual.data != 0)
        np.testing.assert_allclose(actual.toarray(), wanted, atol=2e-15)
    rng = np.random.default_rng(19)
    v = rng.normal(size=g.n_dof)+1j*rng.normal(size=g.n_dof)
    a, b, d = field_oracle(g, v, c)
    np.testing.assert_allclose(p.A @ v, a, atol=2e-14)
    np.testing.assert_allclose(p.B @ v, b, atol=2e-15)
    np.testing.assert_allclose(p.D @ v, d, atol=2e-14)
    # A/B is genuinely a non-Hermitian generalized pencil, not an SPD mass.
    if nx > 1 or ny > 1:
        assert not np.array_equal(p.B.toarray(), p.B.toarray().conj().T)


@pytest.mark.parametrize('nx,ny', [(1, 1), (2, 3), (1, 7), (8, 1), (13, 9)])
@pytest.mark.parametrize('hermitian', [False, True])
def test_03_exact_outgoing_power_identity(nx, ny, hermitian):
    g = StaggeredGrid(nx, ny, 2.7, 5.3)
    c = complex_c()
    if hermitian:
        c = (c+c.conj().T)/2
    p = assemble_pencil(g, c)
    rng = np.random.default_rng(718)
    for _ in range(3):
        v = rng.normal(size=g.n_dof)+1j*rng.normal(size=g.n_dof)
        u = p.B @ v
        cu = (c @ u.reshape(4, -1)).ravel()
        flux = 2*g.hx_um*g.hy_um*np.vdot(u, p.A@v-cu).imag
        f = g.unpack(v)
        edge = (g.hy_um*np.sum(abs(f.Rx[:, -1])**2+abs(f.Sx[:, 0])**2)
                + g.hx_um*np.sum(abs(f.Ry[-1, :])**2+abs(f.Sy[0, :])**2))
        assert flux == pytest.approx(edge, rel=3e-14, abs=3e-13)
        if hermitian:
            assert 2*g.hx_um*g.hy_um*np.vdot(u, p.A@v).imag == pytest.approx(edge, rel=3e-14)
        assert flux >= 0


def manufactured(g, degree):
    """BC-compatible polynomial envelopes; exact analytic center values."""
    amps = np.array([1+2j, 2-1j, -.5+1j, 3+.7j])
    full, exact, deriv = [], [], []
    xc, yc = g.coordinates()
    for k, name in enumerate(NAMES):
        x, y = g.coordinates(name)
        t = x if k < 2 else y
        tc = xc if k < 2 else yc
        length = g.Lx_um if k < 2 else g.Ly_um
        t = t if k % 2 == 0 else length-t
        tc = tc if k % 2 == 0 else length-tc
        full.append(amps[k]*t**degree)
        exact.append(amps[k]*tc**degree)
        # Signed derivative: both outgoing directions have the same sign.
        deriv.append(amps[k]*degree*tc**(degree-1))
    return WaveFields(*full), np.array(exact).reshape(4, -1), np.array(deriv).reshape(4, -1)


def test_04_bc_compatible_linear_and_zero_constant_exact():
    g = StaggeredGrid(7, 5, 3., 8.)
    c = complex_c()
    p = assemble_pencil(g, c)
    f, u, d = manufactured(g, 1)
    v = g.pack(f)
    np.testing.assert_allclose(p.B@v, u.ravel(), atol=4e-15)
    np.testing.assert_allclose(p.A@v, (c@u+1j*d).ravel(), atol=3e-14)
    # A spatially constant field compatible with each incoming zero is zero.
    z = g.pack(WaveFields(*(np.zeros(g.component_shape(n)) for n in NAMES)))
    np.testing.assert_array_equal(p.A@z, z)
    np.testing.assert_array_equal(p.B@z, z)


def convergence_errors():
    errors = []
    for n in (8, 16, 32):
        g = StaggeredGrid(n, 2*n, 2.3, 5.1)
        c = complex_c()
        p = assemble_pencil(g, c)
        f, u, d = manufactured(g, 3)
        v = g.pack(f)
        errors.append((np.linalg.norm(p.B@v-u.ravel())/np.linalg.norm(u),
                       np.linalg.norm(p.D@v-d.ravel())/np.linalg.norm(d),
                       np.linalg.norm(p.A@v-(c@u+1j*d).ravel())/np.linalg.norm(c@u+1j*d)))
    return np.array(errors)


def test_05_cubic_continuous_center_second_order_not_discrete_self_test():
    errors = convergence_errors()
    assert np.all(errors > 1e-7)
    rates = np.log2(errors[:-1]/errors[1:])
    assert np.all((rates > 1.95) & (rates < 2.06)), (errors, rates)


def test_06_linear_storage_and_no_dense_or_inverse(monkeypatch):
    import cwt3d.finite.assembly as assembly
    def forbidden(*args, **kwargs):
        raise AssertionError('dense or inverse/solver operation forbidden in assembly')
    for name in ('inv', 'solve', 'eig', 'eigh'):
        monkeypatch.setattr(np.linalg, name, forbidden)
    for cls in (sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix):
        monkeypatch.setattr(cls, 'toarray', forbidden)
    monkeypatch.setattr(sparse, 'kron', forbidden)
    for name in ('inv', 'spsolve', 'splu', 'eigs', 'eigsh'):
        monkeypatch.setattr(sparse.linalg, name, forbidden)
    monkeypatch.setattr(np, 'eye', forbidden)
    # Guard allocation shape, including all square n_dof-by-n_dof arrays.
    for name in ('zeros', 'empty', 'ones', 'full'):
        original = getattr(np, name)
        def guarded(shape, *args, _original=original, **kwargs):
            if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                assert max(shape) <= 4, f'dense 2-D allocation: {shape}'
            return _original(shape, *args, **kwargs)
        monkeypatch.setattr(np, name, guarded)
    c = complex_c()
    for nx, ny in ((8, 11), (16, 22), (32, 44)):
        g = StaggeredGrid(nx, ny, 2, 3)
        p = assemble_pencil(g, c)
        count = 8*nx*ny-2*nx-2*ny
        assert p.B.nnz == p.D.nnz == count
        assert p.A.nnz == 4*count
        assert p.A.shape == p.B.shape == (4*nx*ny,)*2
        for matrix in (p.A, p.B, p.D):
            storage = matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
            assert storage < 1000*g.n_dof
    source = inspect.getsource(assembly)
    assert 'np.kron(' not in source and 'np.linalg.inv(' not in source


@pytest.mark.filterwarnings('ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning')
def test_07_input_and_public_sparse_snapshot_isolation():
    c = complex_c()
    original = c.copy()
    g = StaggeredGrid(2, 3, 7, 9)
    p = assemble_pencil(g, c)
    before = [m.toarray() for m in (p.A, p.B, p.D)]
    c[:] = 0
    np.testing.assert_array_equal(p.C, original)
    c_out = p.C
    assert not c_out.flags.writeable
    c_out.shape = (16,)
    for name in ('A', 'B', 'D'):
        for mutation in ('data', 'indices', 'indptr', 'shape'):
            public = getattr(p, name)
            if mutation == 'shape':
                public.resize((1, g.n_dof*g.n_dof))
            else:
                getattr(public, mutation)[:] = 0
        np.testing.assert_array_equal(getattr(p, name).toarray(), before[('A', 'B', 'D').index(name)])
    assert p.grid is g
    assert p.ordering == NAMES and p.units == 'um^-1' and p.B_units == '1'
    assert p.eigenvalue_convention == 'lambda = delta + i alpha'
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        p.grid = StaggeredGrid(1, 1, 1, 1)


@pytest.mark.parametrize('c', [np.zeros((3, 3)), np.zeros((4, 4, 1)), np.full((4, 4), np.nan), np.full((4, 4), np.inf), [['bad']*4]*4, None])
def test_08_reject_invalid_c(c):
    with pytest.raises((TypeError, ValueError)):
        assemble_pencil(StaggeredGrid(1, 1, 1, 1), c)


def test_09_reject_grid_and_unrepresentable_indices_before_allocation():
    with pytest.raises(TypeError, match='StaggeredGrid'):
        assemble_pencil((2, 3, 1, 1), np.eye(4))
    lim = int(np.iinfo(np.intp).max)
    for nx, ny in ((lim, 1), (1, lim), (int(np.sqrt(lim)), int(np.sqrt(lim)))):
        g = StaggeredGrid(nx, ny, 1, 1)
        with pytest.raises(ValueError, match='index|representable'):
            assemble_pencil(g, np.eye(4))
    with pytest.raises(ValueError, match='finite|representable'):
        assemble_pencil(StaggeredGrid(1, 1, np.nextafter(0., 1.), 1), np.eye(4))


@pytest.mark.parametrize('empty', [False, True])
def test_10_actual_coupling_result_interface(empty):
    from cwt3d.coupling import build_coupling
    from cwt3d.geometry import CircleHole
    from cwt3d.waveguide import TEWaveguide
    # Existing physical coupling API, not a fake object with invented C_total.
    guide = TEWaveguide([1., 12., 1.], [.2], wavelength_um=1.)
    mode = guide.solve_mode()
    a = 2*np.pi/mode.beta
    fraction = np.pi*.13**2
    holes = {} if empty else {1: CircleHole(.13*a, (a, a),
                                            eps_background=12+2*fraction,
                                            eps_hole=12-2*(1-fraction))}
    result = build_coupling(mode, a, holes, cutoff=1, quadrature_order=6)
    g = StaggeredGrid(2, 3, 4, 9)
    p = assemble_pencil(g, result)
    np.testing.assert_array_equal(p.C, result.C)
    if empty:
        np.testing.assert_array_equal(p.A.toarray(), 1j*p.D.toarray())
    else:
        assert np.linalg.norm(result.C) > 1e-5
        np.testing.assert_allclose(p.A.toarray(), dense_oracle(2, 3, 4, 9, result.C)[0])
    from dataclasses import replace
    with pytest.raises(ValueError, match='units'):
        assemble_pencil(g, replace(result, units='m^-1'))


def test_11_clean_core_import_without_legacy_solver():
    code = """import sys
from cwt3d.finite.assembly import assemble_pencil
from cwt3d.finite import StaggeredGrid
p = assemble_pencil(StaggeredGrid(2,3,4,5), [[0]*4]*4)
assert not any(x.split('.')[0] in ('mph', 'numba', 'model', 'calculator', 'utils') for x in sys.modules)
print(p.A.shape, p.A.nnz, p.B.nnz)
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_12_exact_cancellation_removed_and_finite_overflow_rejected():
    p = assemble_pencil(StaggeredGrid(1, 1, 2, 5),
                        np.diag([-1j, -1j, -.4j, -.4j]))
    assert p.A.nnz == 0
    assert p.A.has_canonical_format
    with pytest.raises(ValueError, match='finite'):
        assemble_pencil(StaggeredGrid(1, 1, 1e-308, 1),
                        np.eye(4)*1.7e308j)
