"""Independent interface-amplitude oracle, never production DtN helpers."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from cwt3d.greens import CutoffError, HomogeneousGreen, LayeredGreen, PoleError
from cwt3d.waveguide import TEWaveguide


def interface_oracle(eps, widths, k0, kp, z, source, derivative=False):
    # Independent small dense system in exp(+/-gamma*z), valid moderate widths.
    edges = np.r_[0., np.cumsum(widths)]
    cuts = np.unique(np.r_[edges, source])
    mids = np.r_[cuts[0]-1, (cuts[:-1]+cuts[1:])/2, cuts[-1]+1]
    material = np.searchsorted(edges, mids, side='right')
    q = np.sqrt(np.asarray(kp**2-k0**2*np.asarray(eps), complex))
    n = len(mids)
    a = np.zeros((2*n, 2*n), complex)
    b = np.zeros(2*n, complex)
    a[0, 1] = 1  # no growing wave in left halfspace
    a[1, -2] = 1  # no growing wave in right halfspace
    for j, x in enumerate(cuts):
        row = 2+2*j
        for region, sign in ((j, -1), (j+1, 1)):
            g = q[material[region]]
            v = np.exp(np.array([g, -g])*x)
            a[row, 2*region:2*region+2] = sign*v
            a[row+1, 2*region:2*region+2] = sign*v*np.array([g, -g])
        b[row+1] = -1 if x == source else 0
    c = np.linalg.solve(a, b)
    result = []
    for x in np.atleast_1d(z):
        j = np.searchsorted(cuts, x, side='right')
        g = q[material[j]]
        basis = np.exp(np.array([g, -g])*x)
        if derivative:
            basis *= np.array([g, -g])
        result.append(basis @ c[2*j:2*j+2])
    return np.array(result)


@pytest.mark.parametrize('eps,kp', [(2.3, 0), (2.3-.2j, .4), (1, 4)])
def test_uniform_partition(eps, kp):
    g = LayeredGreen(1.2, [eps]*4, [.4, .7], kp)
    h = HomogeneousGreen(1.2, eps, kp)
    z = np.array([-1., 0., .2, .4, .8, 1.1, 2.])[:, None]
    s = np.array([-.3, 0., .3, .4, 1.1, 1.6])[None, :]
    assert_allclose(g(z, s), h(z, s), rtol=2e-13, atol=2e-14)
    assert_allclose(g.derivative(z, s, side='right'), h.derivative(z, s, side='right'), atol=3e-14)


@pytest.mark.parametrize('source', [-.4, 0., .3])
def test_single_interface_fresnel(source):
    e = [2.2-.1j, 4.1-.3j]
    g = LayeredGreen(1.3, e, [], .7)
    q = np.sqrt(.7**2-1.3**2*np.array(e))
    z = np.array([-.9, -.1, .1, .8])
    j = 0 if source < 0 else 1
    same = z < 0 if j == 0 else z >= 0
    expected = np.empty(z.size, complex)
    expected[same] = (np.exp(-q[j]*abs(z[same]-source)) +
        (q[j]-q[1-j])/(q[j]+q[1-j])*np.exp(-q[j]*(abs(z[same])+abs(source))))/(2*q[j])
    expected[~same] = np.exp(-q[j]*abs(source)-q[1-j]*abs(z[~same]))/(q[0]+q[1])
    assert_allclose(g(z, source), expected, atol=2e-14)


@pytest.mark.parametrize('eps,kp', [([1, 5, 2, 3, 1.5], .6),
    ([1-.1j, 5-.3j, 2-.2j, 3-.5j, 1.5-.1j], 1.7)])
def test_multilayer_independent_oracle_reciprocity(eps, kp):
    widths = [.3, .4, .2]
    g = LayeredGreen(1.4, eps, widths, kp)
    z = np.array([-.4, 0., .12, .3, .5, .7, .8, .9, 1.2])
    for s in [-.3, 0., .2, .3, .55, .9, 1.3]:
        assert_allclose(g(z, s), interface_oracle(eps, widths, 1.4, kp, z, s), atol=5e-13)
        mask = z != s
        assert_allclose(g.derivative(z[mask], s), interface_oracle(eps, widths, 1.4, kp, z[mask], s, True), atol=1e-12)
    matrix = g(z[:, None], z[None, :])
    assert_allclose(matrix, matrix.T, atol=3e-14)
    if np.iscomplexobj(eps):
        assert np.max(abs(matrix-matrix.conj().T)) > .01


def test_physical_ode_interfaces_source_jump_flux():
    g = LayeredGreen(1.3, [1-.1j, 4-.2j, 2-.1j, 1.5-.2j], [.4, .6], .7)
    z = np.array([-.3, .13, .3, .6, .85, 1.3])
    s = .22
    h = 1e-4
    potential = (g.k0**2*g.eps_profile(z)-g.k_parallel**2)*g(z, s)
    residual = ((g(z+h,s)-2*g(z,s)+g(z-h,s))/h**2+potential)
    assert np.max(abs(residual)/abs(potential)) < 5e-7
    h = 1e-7
    assert_allclose((g(z+h,s)-g(z-h,s))/(2*h), g.derivative(z,s), atol=1e-8)
    for x in g.interfaces_um:
        assert_allclose(g(x-h,s), g(x+h,s), atol=3e-7)
        assert_allclose(g.derivative(x-h,s), g.derivative(x+h,s), atol=4e-7)
    for s in [-.3, 0., .2, .4, 1., 1.2]:
        assert_allclose(g.derivative(s,s,side='right')-g.derivative(s,s,side='left'), -1, atol=2e-14)
        slopes = (g(s+h,s)-2*g(s,s)+g(s-h,s))/h
        assert abs(slopes+1) < 2e-6
    for x in [-2., 3.]:
        flux = -np.imag(np.conj(g(x,.2))*g.derivative(x,.2))
        assert flux*np.sign(x) > 0
    assert abs(g(100,.2)) < abs(g(3,.2))


def test_internal_cutoff_linear_limit():
    g = LayeredGreen(1, [2, 1, 3], [.7], 1)
    z = np.linspace(.1,.6,6)
    assert_allclose(np.diff(g(z,-.2), n=2), 0, atol=1e-14)
    assert_allclose(g.derivative(z,-.2), np.repeat(g.derivative(.3,-.2),6), atol=1e-14)
    near = LayeredGreen(1,[2,1-1e-9,3],[.7],1)
    assert_allclose(g(z,-.2), near(z,-.2), atol=1e-9)


def test_thick_barrier_and_distant_points_strict():
    with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
        g = LayeredGreen(1, [1, 2, 1], [10000.], 3)
        q0,q1 = np.sqrt(8),np.sqrt(7)
        expected = (1+(q0-q1)/(q0+q1)*np.exp(-q0*.4))/(2*q0)
        assert_allclose(g(-.2,-.2), expected, atol=1e-14)
        assert g(10001.,-1.) == 0
        assert g(1e6,-1e6) == 0
        assert g.derivative(1e6,-1e6) == 0
        assert_allclose(g(5000.,5000.), 1/(2*q1), atol=1e-14)
        # Two individually representable tails can have an unrepresentable
        # product. This is propagation underflow, not a physical failure.
        assert g(400.,800.) == 0
        assert g.derivative(400.,800.) == 0
        assert g(-150.,150.) == 0
        assert g.derivative(-150.,150.) == 0


def test_true_guided_pole_and_nonzero_near_pole():
    guide = TEWaveguide([1,9,1],[.6],k0=4)
    beta = guide.solve_mode().beta
    with pytest.raises(PoleError, match='pole|singular'):
        LayeredGreen(4, guide.eps, guide.thickness_um, beta)(.2,.3)
    vals = []
    for offset in [1e-4, 1e-5, 1e-6]:
        g = LayeredGreen(4,guide.eps,guide.thickness_um,beta+offset)
        actual = g(.2,.3)
        expected = interface_oracle(guide.eps,guide.thickness_um,4,beta+offset,[.2],.3)[0]
        assert_allclose(actual,expected,rtol=2e-8)
        vals.append(abs(actual))
    assert_allclose(np.array(vals[1:])/vals[:-1], 10, rtol=2e-4)


def test_scaling_translation_and_structure_snapshot():
    eps = np.array([1-.1j,4-.2j,2-.3j])
    d = np.array([.7])
    g = LayeredGreen(1.3, eps, d, .4, origin_um=2)
    h = LayeredGreen(1.3,eps,d,.4)
    z,s = np.array([-.3,.2,1.]),.1
    assert_allclose(g(z+2,s+2),h(z,s),atol=1e-14)
    scale = 3.7
    scaled = LayeredGreen(1.3/scale,eps,d*scale,.4/scale)
    assert_allclose(scaled(z*scale,s*scale),scale*h(z,s),atol=1e-14)
    assert_allclose(scaled.derivative(z*scale,s*scale),h.derivative(z,s),atol=1e-14)
    expected = g(z,s)
    eps[:] = 8; d[:] = 2
    for name in ['eps','thickness_um','interfaces_um','k0','k_parallel','origin_um']:
        with pytest.raises(AttributeError): setattr(g,name,0)
        with pytest.raises(AttributeError): delattr(g,name)
    out = g(z,s); out[:] = 100
    assert_allclose(g(z,s),expected,atol=0,rtol=0)
    assert isinstance(g.eps,tuple) and isinstance(g.thickness_um,tuple)
    assert g([],0).shape == (0,)
    guide = TEWaveguide([1,4,1],[.7],k0=1.3)
    assert_allclose(LayeredGreen.from_structure(guide,.4)(z,s), LayeredGreen(1.3,[1,4,1],[.7],.4)(z,s))


@pytest.mark.parametrize('kwargs', [dict(k0=0),dict(k0=True),dict(k_parallel=-1),
    dict(eps=[1,2+.1j,1]),dict(eps=[1,np.nan,1]),dict(eps=[True,2,1]),
    dict(thickness_um=[0]),dict(thickness_um=[]),dict(origin_um=np.inf)])
def test_invalid_configuration(kwargs):
    config = dict(k0=1,eps=[1,2,1],thickness_um=[.5])
    config.update(kwargs)
    with pytest.raises(ValueError): LayeredGreen(**config)


def test_errors_leave_object_and_seterr_intact():
    g = LayeredGreen(1,[1,3,2],[.5],.3)
    before = g(.2,.3)
    policy = np.geterr()
    for bad in [True,1j,np.nan,'1',[0,True]]:
        with pytest.raises(ValueError): g(bad,0)
    for side in [True,0,[], 'average']:
        with pytest.raises(ValueError): g.derivative(.1,.2,side=side)
    with pytest.raises(ValueError,match='source'): g.derivative(.2,.2)
    assert np.geterr() == policy
    assert g(.2,.3) == before


def test_random_passive_and_negative_materials_independent_oracle():
    rng = np.random.default_rng(20261025)
    for _ in range(15):
        eps = rng.uniform(-2, 6, 5)-1j*rng.uniform(.01,.6,5)
        widths = rng.uniform(.1,.5,3)
        k0,kp = rng.uniform(.7,2),rng.uniform(0,2)
        g = LayeredGreen(k0,eps,widths,kp)
        z = np.linspace(-.4,sum(widths)+.4,7)
        for source in [-.2,.15,sum(widths)+.2]:
            assert_allclose(g(z,source),interface_oracle(eps,widths,k0,kp,z,source),rtol=2e-12,atol=2e-13)


def test_element_dirichlet_resonance_is_not_global_pole():
    # q*d=i*pi makes an unsplit endpoint map singular, while this radiating
    # physical structure has a perfectly finite Green function.
    g = LayeredGreen(1,[1,4,1],[np.pi/2])
    z = np.array([-.2,.1,.7,1.3,2.])
    assert_allclose(g(z,.3),interface_oracle([1,4,1],[np.pi/2],1,0,z,.3),atol=3e-14)


def test_lossy_thick_layer_and_strict_intermediate_underflow():
    with np.errstate(over='raise',invalid='raise',divide='raise',under='raise'):
        eps = 2-3j
        g = LayeredGreen(1,[eps]*3,[10000.],3)
        h = HomogeneousGreen(1,eps,3)
        # Homogeneous oracle at modest separations only; its unchanged API
        # intentionally obeys caller underflow policy at very long distances.
        assert_allclose(g(5000.1,5000.),h(5000.1,5000.),atol=1e-14)
        assert g(10001.,-1.) == 0
        assert g.derivative(10001.,-1.) == 0


def test_exterior_cutoff_and_resource_limits_are_explicit():
    with pytest.raises(CutoffError,match='exterior cutoff'):
        LayeredGreen(1,[1,4,2],[.7],1)
    with pytest.raises(ValueError,match='budget'):
        LayeredGreen(1,[1,4,1],[10000.])


@pytest.mark.filterwarnings('ignore:Setting the shape on a NumPy array:DeprecationWarning')
def test_broadcast_metadata_and_profile_are_isolated():
    g = LayeredGreen(1,[1,4,2],[.5],.2)
    z = np.array([-.2,0.,.3,.5,.8])[:,None]
    s = np.array([-.1,.2,.5])[None,:]
    z0,s0 = z.copy(),s.copy()
    expected = g(z,s)
    out = g(z,s)
    out.shape = (15,)
    out[:] = 0
    profile = g.eps_profile(z)
    profile.shape = (5,)
    profile[:] = 100
    assert_allclose(g.eps_profile([0.,.5]),[4,2])
    assert_allclose(g(z,s),expected,rtol=0,atol=0)
    assert_allclose(z,z0); assert_allclose(s,s0)
    assert g(np.ones((2,1,3)), np.zeros((4,1))).shape == (2,4,3)
    with pytest.raises(ValueError): g(np.zeros(2),np.zeros(3))

