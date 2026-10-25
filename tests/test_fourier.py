import numpy as np
import pytest

from cwt3d.geometry import CircleHole, RightTriangleHole, SampledPermittivity, sample_permittivity
from cwt3d.fourier import shape_fourier, integrate_hole_fourier, integrate_fourier, fft_fourier

ORDERS = np.array([[0,0], [1,0], [-1,0], [0,2], [2,-3], [-2,3]])


def test_circle_analytic_independent_polar_and_translation():
    c = CircleHole(.21, period_um=(1,1.3), center_um=(.43,.61), eps_background=3+2j, eps_hole=1-.2j)
    analytic = shape_fourier(c, ORDERS)
    polar = integrate_hole_fourier(c, ORDERS, quadrature_order=80)
    np.testing.assert_allclose(analytic, polar, atol=2e-13)
    assert analytic[0] == pytest.approx(c.mean_epsilon)
    shift = np.array([.03,-.04])
    moved = CircleHole(.21, period_um=c.period_um, center_um=np.array(c.center_um)+shift, eps_background=c.eps_background, eps_hole=c.eps_hole)
    phase = np.exp(2j*np.pi*(ORDERS*shift/np.array(c.period_um)).sum(axis=1))
    np.testing.assert_allclose(shape_fourier(moved, ORDERS), analytic*phase, atol=2e-13)
    assert abs(analytic[1]-analytic[2].conjugate()) > .01


def test_triangle_duffy_convergence_and_average():
    t = RightTriangleHole((.61,.37), center_um=(.45,.5), eps_background=4+.2j, eps_hole=1-.5j)
    orders = np.vstack([ORDERS, [13,-11]])
    low = shape_fourier(t, orders, quadrature_order=8)
    medium = shape_fourier(t, orders, quadrature_order=48)
    high = shape_fourier(t, orders, quadrature_order=64)
    assert high[0] == pytest.approx(t.mean_epsilon, abs=2e-14)
    assert np.max(abs(medium-high)) < 2e-13
    assert np.max(abs(low-high)) > 1e-6
    # Independent full-cell midpoint sampling, not the Duffy mapping.
    errors = []
    for shape in [(451,550), (901,1100), (1801,2200)]:
        sampled = fft_fourier(sample_permittivity(t, t.period_um, shape), ORDERS)
        errors.append(np.max(abs(sampled-high[:len(ORDERS)])))
    # Discontinuous boundaries need not converge monotonically per grid.
    assert errors[-1] < errors[0]/2
    assert errors[-1] < 6e-4


def smooth(x,y):
    return 2+.7j + (1+.3j)*np.exp(-2j*np.pi*(2*x/1.2-y/.9)) + (.2-.6j)*np.exp(-2j*np.pi*(-x/1.2+3*y/.9))


def test_fft_reference_non_square_sign_axes_reconstruction_first_repeat():
    period = (1.2,.9)
    orders = np.array([[0,0],[2,-1],[-1,3],[-2,1],[1,-3]])
    grid = sample_permittivity(smooth, period, (35,48))
    expected = np.array([2+.7j,1+.3j,.2-.6j,0,0])
    first = fft_fourier(grid, orders)
    np.testing.assert_allclose(first, expected, atol=2e-14)
    np.testing.assert_array_equal(fft_fourier(grid, orders), first)
    np.testing.assert_allclose(integrate_fourier(smooth, period, orders, quadrature_order=64), expected, atol=4e-14)
    x = (np.arange(48)+.5)*period[0]/48
    y = (np.arange(35)+.5)*period[1]/35
    rebuilt = sum(c*np.exp(-2j*np.pi*(m*x[None,:]/period[0]+n*y[:,None]/period[1])) for c,(m,n) in zip(first,orders))
    np.testing.assert_allclose(rebuilt, grid.values, atol=3e-14)
    shift = (.071,-.042)
    moved = sample_permittivity(lambda x,y: smooth(x-shift[0],y-shift[1]), period, (35,48))
    phase = np.exp(2j*np.pi*(orders*np.array(shift)/period).sum(axis=1))
    np.testing.assert_allclose(fft_fourier(moved,orders), first*phase, atol=2e-14)
    # Different data on subsequent call must never reuse a stale spectrum.
    np.testing.assert_allclose(fft_fourier(SampledPermittivity(2*grid.values,period),orders),2*first,atol=2e-14)


@pytest.mark.parametrize('metadata', ['shape', 'dtype', 'strides'])
def test_sampled_metadata_cannot_change_snapshot_or_fft(metadata):
    source = np.arange(12).reshape(3, 4)
    original = source.copy()
    source_metadata = (source.shape, source.dtype, source.strides)
    sampled = SampledPermittivity(source)
    retained = sampled.values
    independent = SampledPermittivity(retained)
    orders = [[0, 0], [1, 0], [-1, 0], [0, -1]]
    before = fft_fourier(sampled, orders)

    # Exercise every ndarray reachable through the public base chain, not
    # just the returned view: a view of a private ndarray still leaks metadata.
    exposed = sampled.values
    targets = []
    while isinstance(exposed, np.ndarray):
        targets.append(exposed)
        exposed = exposed.base
    assert isinstance(exposed, bytes)
    for target in targets:
        assert not target.flags.writeable
        with pytest.raises(ValueError):
            target.setflags(write=True)
        with pytest.raises(ValueError):
            target.flat[0] = 99
        if metadata == 'shape':
            target.shape = (2, 6)
        elif metadata == 'dtype':
            target.dtype = np.float64
        else:
            target.strides = ((0, 16) if target.ndim == 2 else (0,))

        np.testing.assert_array_equal(fft_fourier(sampled, orders), before)
        for snapshot in (sampled.values, retained, independent.values):
            assert snapshot.shape == (3, 4)
            assert snapshot.dtype == np.dtype(complex)
            assert snapshot.strides == (64, 16)
            assert not snapshot.flags.writeable
            np.testing.assert_array_equal(snapshot, original)
        assert (source.shape, source.dtype, source.strides) == source_metadata
        np.testing.assert_array_equal(source, original)


def test_discontinuous_circle_resolution_convergence():
    c = CircleHole(.217, center_um=(.437,.519))
    exact = shape_fourier(c,ORDERS)
    errors = []
    for ny,nx in [(41,54),(181,246),(701,950)]:
        errors.append(np.max(abs(fft_fourier(sample_permittivity(c,c.period_um,(ny,nx)),ORDERS)-exact)))
    assert errors[-1] < errors[0]/5
    assert errors[-1] < 2e-4


def test_real_only_conjugate_symmetry():
    g = sample_permittivity(lambda x,y: np.cos(2*np.pi*(x+2*y))+.3*np.sin(2*np.pi*x), (1,1),(21,32))
    result = fft_fourier(g, [[1,2],[-1,-2],[1,0],[-1,0]])
    np.testing.assert_allclose(result[[1,3]], result[[0,2]].conjugate(), atol=1e-15)


def test_fft_random_grid_all_canonical_orders_against_explicit_sum():
    rng = np.random.default_rng(37)
    ny,nx = 7,10
    values = rng.normal(size=(ny,nx))+1j*rng.normal(size=(ny,nx))
    orders = np.array([(m,n) for n in range(-(ny//2),(ny-1)//2+1)
                       for m in range(-(nx//2),(nx-1)//2+1)])
    x = (np.arange(nx)+.5)/nx
    y = (np.arange(ny)+.5)/ny
    expected = [np.mean(values*np.exp(2j*np.pi*(m*x[None,:]+n*y[:,None])))
                for m,n in orders]
    np.testing.assert_allclose(fft_fourier(SampledPermittivity(values),orders), expected, atol=2e-15)


def test_triangle_translation_and_order_shape():
    shift = np.array([.02,-.03])
    t = RightTriangleHole((.3,.4),center_um=(.5,.5))
    moved = RightTriangleHole(t.legs_um,center_um=np.array(t.center_um)+shift)
    orders = ORDERS.reshape(2,3,2)
    phase = np.exp(2j*np.pi*np.sum(orders*shift,axis=-1))
    np.testing.assert_allclose(shape_fourier(moved,orders),shape_fourier(t,orders)*phase,atol=2e-15)


@pytest.mark.parametrize('orders', [[[1.2,0]], [[True,False]], [[1]], [[np.nan,0]]])
def test_bad_orders(orders):
    with pytest.raises((TypeError,ValueError)):
        fft_fourier(SampledPermittivity(np.ones((5,8))), orders)


@pytest.mark.parametrize('orders', [[[4,0]], [[0,3]], [[-5,0]], [[0,-3]], [[1,0],[1,0]], [[-4,0],[4,0]]])
def test_nyquist_or_duplicate_rejected(orders):
    with pytest.raises(ValueError):
        fft_fourier(SampledPermittivity(np.ones((5,8))),orders)


def test_even_negative_nyquist_center_phase():
    values = np.tile((-1.)**np.arange(8),(5,1))
    np.testing.assert_allclose(fft_fourier(SampledPermittivity(values),[[-4,0]]),[-1j],atol=1e-15)


def test_integrator_rejects_nonbroadcast_callable_output():
    with pytest.raises(ValueError):
        integrate_fourier(lambda x,y: np.ones((2,16,16)), (1,1), [0,0], quadrature_order=16)


@pytest.mark.parametrize('q', [0,2.5,True])
def test_bad_quadrature(q):
    with pytest.raises((TypeError,ValueError)):
        integrate_fourier(smooth,(1.2,.9),ORDERS,quadrature_order=q)


def test_scalar_order_and_constant_callable():
    assert fft_fourier(SampledPermittivity(np.ones((3,4))),[0,0]).shape == ()
    assert integrate_fourier(lambda x,y: 3+2j,(1,1),[0,0]) == pytest.approx(3+2j)
    with pytest.raises(ValueError):
        integrate_fourier(lambda x,y: np.nan,(1,1),[0,0])
