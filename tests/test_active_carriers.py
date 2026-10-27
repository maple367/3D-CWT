import numpy as np
import pytest
from scipy import sparse
from scipy.constants import elementary_charge

from cwt3d.active.materials import CarrierMaterial
from cwt3d.active.carriers import carrier_rhs, diffusion_operator
from cwt3d.finite.grid import StaggeredGrid


@pytest.mark.parametrize('nx,ny', [(3, 2), (1, 3), (4, 1), (1, 1)])
def test_diffusion_finite_volume(nx, ny):
    grid = StaggeredGrid(nx, ny, 2*nx, 3*ny)
    d = diffusion_operator(grid, 6.)
    assert sparse.issparse(d)
    assert d.shape == (nx*ny, nx*ny)
    expected = np.zeros(d.shape)
    for y in range(ny):
        for x in range(nx):
            i = y*nx+x
            for yy, xx, rate in [(y, x+1, 6/4), (y+1, x, 6/9)]:
                if yy < ny and xx < nx:
                    j = yy*nx+xx
                    expected[i, j] += rate
                    expected[j, i] += rate
                    expected[i, i] -= rate
                    expected[j, j] -= rate
    np.testing.assert_allclose(d.toarray(), expected)
    np.testing.assert_allclose(d @ np.ones(nx*ny), 0., atol=1e-14)
    n = np.arange(nx*ny, dtype=float)**2
    assert abs(np.sum(d @ n)) < 1e-12
    assert n @ (d @ n) <= 1e-12
    assert np.linalg.eigvalsh(d.toarray()).max() <= 1e-12
    assert diffusion_operator(grid, 0.).nnz == 0


def rhs(n, u=0., j=0., material=None, **kwargs):
    return carrier_rhs(n, u, j, material or CarrierMaterial(1., 0.),
                       vg_um_ns=kwargs.pop('vg_um_ns', 2.),
                       active_thickness_um=kwargs.pop('active_thickness_um', .5),
                       diffusion_matrix=kwargs.pop('diffusion_matrix', sparse.csr_matrix((np.size(n),)*2)),
                       **kwargs)


def test_current_conversion_recombination_and_shape():
    n = np.array([[1., 2.], [3., 4.]])
    m = CarrierMaterial(1., 0., recombination_a_ns=2., recombination_b_um3_ns=3.,
                        recombination_c_um6_ns=4.)
    j = elementary_charge*.5*1e9
    np.testing.assert_allclose(rhs(n, j=j, material=m), 1.-2*n-3*n*n-4*n*n*n)
    assert rhs(n).shape == n.shape
    assert rhs(n.ravel()).shape == (4,)
    assert np.shape(rhs(2.)) == ()


def test_signed_absorption_and_shared_compression():
    n = np.array([[-1., 2.]])
    u = np.array([[3., 4.]])
    m = CarrierMaterial(2., 1., gain_compression_um3=.25)
    np.testing.assert_allclose(rhs(n, u, material=m), -2*m.gain(n, u)*u)
    assert rhs(n, u, material=m)[0, 0] > 0  # absorbed photons generate carriers


def test_rhs_diffusion_only_conservation():
    grid = StaggeredGrid(3, 2, 5., 7.)
    n = np.arange(6.).reshape(2, 3)
    d = diffusion_operator(grid, 2.)
    np.testing.assert_allclose(rhs(n, diffusion_matrix=d).ravel(), d @ n.ravel())
    assert abs(rhs(n, diffusion_matrix=d).sum()) < 1e-13


@pytest.mark.parametrize('bad', [-1., True, 1j, np.nan, np.inf, [1.]])
def test_bad_diffusion(bad):
    with pytest.raises(ValueError):
        diffusion_operator(StaggeredGrid(2, 2, 1., 1.), bad)


@pytest.mark.parametrize('key,bad', [
    ('j', -1.), ('j', True), ('j', np.nan), ('j', np.ones((1, 2))),
    ('u', -1.), ('u', 1j), ('u', np.ones((2, 1))),
    ('vg_um_ns', 0.), ('vg_um_ns', True), ('vg_um_ns', np.inf),
    ('active_thickness_um', -1.), ('active_thickness_um', [1.]),
    ('diffusion_matrix', sparse.eye(3)),
    ('diffusion_matrix', sparse.eye(4, dtype=complex)),
    ('diffusion_matrix', sparse.csr_matrix(np.full((4, 4), np.nan))),
])
def test_bad_rhs_controls(key, bad):
    with pytest.raises(ValueError):
        rhs(np.ones((2, 2)), **{key: bad})


def test_rhs_material_maps_and_bad_flattening():
    m = CarrierMaterial(np.ones((2, 2)), 0.)
    assert rhs(np.ones((2, 2)), material=m).shape == (2, 2)
    with pytest.raises(ValueError):
        rhs(np.ones(4), material=m)
    with pytest.raises(ValueError):
        rhs(np.ones((2, 1)), material=m)


def test_rhs_overflow_and_nonreal_trials():
    for n in [True, 1j, np.nan, np.inf]:
        with pytest.raises(ValueError):
            rhs(n)
    with pytest.raises(ValueError):
        rhs(1e308, u=1e308)
    with pytest.raises(ValueError):
        rhs(1., j=1e308, active_thickness_um=1e-308)
