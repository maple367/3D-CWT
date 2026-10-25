"""Independent physical checks of the real structure -> four-wave API."""
import numpy as np
import pytest
from scipy.integrate import quad
from cwt3d.geometry import CircleHole
from cwt3d.waveguide import TEWaveguide
from cwt3d.fourier import shape_fourier
from cwt3d.greens import LayeredGreen
from cwt3d.coupling import build_coupling
from cwt3d.infinite import solve_infinite
from cwt3d.quadrature import green_overlap


def physical(scale=1., dual=False, shift=(0., 0.), contrast=2.):
    widths = np.array([.13, .19] if dual else [.32])*scale
    means = [8.,10.] if dual else [9.]
    guide = TEWaveguide([1., *means, 1.], widths, k0=6./scale)
    mode = guide.solve_mode()
    a = 2*np.pi/mode.beta
    holes = {}
    for layer in range(1, len(widths)+1):
        r = (.13 if layer == 1 else .21)*a
        f = np.pi*(r/a)**2
        center = np.array((.36, .47) if layer == 1 else (.63, .54))+shift
        holes[layer] = CircleHole(r, (a,a), tuple(center*a),
                                 means[layer-1]+contrast*f, means[layer-1]-contrast*(1-f))
    return mode, a, holes


def build(scale=1., dual=False, shift=(0.,0.), contrast=2., order=10, cutoff=1):
    m,a,h = physical(scale, dual, shift, contrast)
    return build_coupling(m, a, h, cutoff=cutoff, quadrature_order=order)


def test_physical_symmetry_power_and_four_eigenpairs():
    c = build(dual=True)
    np.testing.assert_allclose(c.C1D, c.C1D.conj().T, atol=2e-13)
    assert c.all_aux_evanescent
    np.testing.assert_allclose(c.C2D, c.C2D.conj().T, atol=2e-12)
    loss = (c.Crad-c.Crad.conj().T)/(2j)
    assert np.linalg.eigvalsh(loss).min() > -1e-12
    v = np.array([1, 2j, -.7+.2j, .3j])
    assert np.imag(np.vdot(v,c.C@v)) == pytest.approx(np.vdot(v,loss@v).real, abs=2e-12)
    modes = solve_infinite(c)
    assert modes.eigenvalues.shape == (4,)
    assert max(modes.residuals) < 1e-14
    np.testing.assert_allclose(modes.power_loss, 2*modes.alpha)


def test_all_terms_inverse_length_including_chi_gate():
    c, scaled = build(), build(scale=3.7)
    for name in ('C1D','Crad','C2D','C2D_green','C2D_local','C'):
        np.testing.assert_allclose(getattr(c,name)/3.7, getattr(scaled,name), rtol=2e-11, atol=2e-14)
    assert np.linalg.norm(c.C2D) > 1e-5


def test_single_layer_kappa_and_chi_against_direct_nested_quad():
    mode,a,h = physical()
    c = build_coupling(mode,a,h,cutoff=1,quadrature_order=12)
    k,b = mode.guide.k0,2*np.pi/a
    d = mode.guide.thickness_um[0]
    overlap = quad(lambda z: abs(mode.theta(z))**2,0,d,epsabs=1e-12)[0]
    xi = lambda p,q: shape_fourier(h[1],np.array([p,q])).item()
    assert c.C1D[0,1] == pytest.approx(-k*k/(2*b)*xi(2,0)*overlap,abs=2e-13)
    # D=1: only four diagonal auxiliary waves. For yy(0,0), transverse
    # projection times A is +m^2; longitudinal projection times -B is -n^2.
    g = LayeredGreen.from_structure(mode.guide,b*np.sqrt(2))
    def inner(z):
        return quad(lambda s: (g(z,s)*mode.theta(s)).real,0,z,epsabs=1e-12)[0]+quad(lambda s: (g(z,s)*mode.theta(s)).real,z,d,epsabs=1e-12)[0]
    K = quad(lambda z: mode.theta(z).real*inner(z),0,d,epsabs=1e-11)[0]
    direct = sum(xi(1-m,-n)*xi(m-1,n)*(k*k*K-overlap/9.)/2
                 for m in (-1,1) for n in (-1,1))
    assert c.C2D[0,0] == pytest.approx(-k*k/(2*b)*direct,rel=2e-10,abs=2e-13)


def test_dual_layer_translation_and_source_not_averaged():
    c = build(dual=True)
    shifted = build(dual=True,shift=(.03,-.04))
    orders = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    phase = np.exp(2j*np.pi*(orders@np.array([.03,-.04])))
    np.testing.assert_allclose(shifted.C, phase[:,None]*c.C/phase[None,:],atol=3e-12)
    np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(shifted.C)),np.sort_complex(np.linalg.eigvals(c.C)),atol=2e-12)
    mode,a,holes = physical(dual=True)
    # Independent source/receiver integral for one radiative matrix entry.
    g = LayeredGreen.from_structure(mode.guide)
    edges = mode.guide.interfaces_um
    xi = lambda l,p,q: shape_fourier(holes[l],np.array([p,q])).item()
    direct = 0j
    wrong = 0j
    for l in (1,2):
        for s in (1,2):
            K = green_overlap(mode,(edges[l-1],edges[l]),(edges[s-1],edges[s]),g,12)
            direct += xi(l,1,0)*xi(s,-1,0)*K
            wrong += xi(l,1,0)*xi(l,-1,0)*K
    pref = -mode.guide.k0**4/(2*(2*np.pi/a))
    assert c.Crad[0,0] == pytest.approx(pref*direct,abs=3e-12)
    assert abs(pref*(wrong-direct)) > 1e-5


def test_quadrature_and_cutoff_convergence():
    a,b,c = (build(order=n) for n in (4,8,12))
    assert np.linalg.norm(b.C-c.C) < 1e-4*np.linalg.norm(a.C-c.C)
    ds = [build(cutoff=d,order=10) for d in (1,2,3,4)]
    assert np.linalg.norm(ds[3].C-ds[2].C) < np.linalg.norm(ds[1].C-ds[0].C)


def test_zero_contrast_empty_and_snapshot_metadata():
    c = build(contrast=0)
    np.testing.assert_array_equal(c.C,0)
    mode,a,_ = physical()
    np.testing.assert_array_equal(build_coupling(mode,a,{}).C,0)
    c = build()
    expected = c.C.copy()
    exposed = c.C
    with pytest.raises(ValueError):
        exposed[0,0] = 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        exposed.shape = (16,)
        exposed.dtype = np.float64
    np.testing.assert_array_equal(c.C,expected)
    eigen = solve_infinite(c)
    assert eigen.eigenvectors.shape == (4,4)
    for result,names in ((c,('C','C1D','Crad','C2D','C2D_green','C2D_local','xi','K','mu','nu','varsigma')),
                         (eigen,('eigenvalues','eigenvectors','delta','alpha','power_loss','residuals'))):
        for name in names:
            a = getattr(result,name)
            before = a.copy()
            assert isinstance(a.base,bytes)
            with pytest.raises(ValueError):
                a.setflags(write=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                a.shape = (a.size,)
                a.strides = (0,)
            np.testing.assert_array_equal(getattr(result,name),before)
    with pytest.raises(AttributeError):
        c.period_um = 1.
    with pytest.raises(AttributeError):
        eigen.units = 'dimensionless'
    np.testing.assert_array_equal(solve_infinite(c).eigenvalues,eigen.eigenvalues)


def test_reject_mismatch_loss_and_bad_inputs():
    mode,a,h = physical()
    with pytest.raises(ValueError,match='beta'):
        build_coupling(mode,1.01*a,h)
    with pytest.raises(ValueError,match='mean'):
        build_coupling(mode,a,{1:CircleHole(.1*a,(a,a),eps_background=10,eps_hole=1)})
    with pytest.raises(ValueError,match='lossless'):
        build_coupling(mode,a,{1:CircleHole(.1*a,(a,a),eps_background=9-.01j,eps_hole=1)})
    for value in (0,True,1.5):
        with pytest.raises((ValueError,TypeError)):
            build_coupling(mode,a,h,cutoff=value)
    with pytest.raises(ValueError):
        build_coupling(mode,a,{0:h[1]})


def cquad(fn,a,b,**kw):
    return quad(lambda z: fn(z).real,a,b,epsabs=1e-11,**kw)[0]+1j*quad(lambda z: fn(z).imag,a,b,epsabs=1e-11,**kw)[0]


def test_independent_dual_layer_radiation_integral_and_outgoing_flux():
    mode,a,holes = physical(dual=True)
    c = build_coupling(mode,a,holes,quadrature_order=12)
    guide = mode.guide
    g = LayeredGreen.from_structure(guide)
    edges = guide.interfaces_um
    basic = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    v = np.array([1+.2j,-.3j,.8,.2-.4j])
    source_coeff = {l:shape_fourier(h,-basic) for l,h in holes.items()}
    def field(z,pol):
        total = 0j
        for l in holes:
            left,right = edges[l-1:l+1]
            points = [z] if left < z < right else None
            amp = source_coeff[l][2*pol:2*pol+2]@v[2*pol:2*pol+2]
            total += amp*cquad(lambda s:g(z,s)*mode.theta(s),left,right,points=points)
        return total
    integral = 0j
    flux = 0.
    for pol in (0,1):
        for l in holes:
            amp = source_coeff[l][2*pol:2*pol+2]@v[2*pol:2*pol+2]
            integral += cquad(lambda z:(amp*mode.theta(z)).conjugate()*field(z,pol),edges[l-1],edges[l])
        flux += guide.k0*(abs(field(0.,pol))**2+abs(field(edges[-1],pol))**2)
    pref = -guide.k0**4/(2*c.beta0)
    matrix_form = np.vdot(v,c.Crad@v)
    assert matrix_form == pytest.approx(pref*integral,rel=2e-10,abs=2e-12)
    assert 2*matrix_form.imag == pytest.approx(guide.k0**4/c.beta0*flux,rel=2e-10,abs=2e-12)


def test_propagating_internal_auxiliary_and_explicit_detuning_tolerance():
    mode = TEWaveguide([1.,9.,1.],[.025],k0=6.).solve_mode()
    a = 2*np.pi/mode.beta
    f = np.pi*.15**2
    hole = CircleHole(.15*a,(a,a),eps_background=9+f,eps_hole=8+f)
    c = build_coupling(mode,a,{1:hole},quadrature_order=12)
    assert not c.all_aux_evanescent
    assert np.isfinite(solve_infinite(c).eigenvalues).all()
    detuned = build_coupling(mode,1.0001*a,{},beta_rtol=.001)
    assert detuned.period_um == 1.0001*a
    assert detuned.relative_detuning == pytest.approx(.0001)


def test_duffy_cusp_against_closed_form_not_tensor_diagonal():
    from cwt3d.greens import HomogeneousGreen
    # Constant normalized field on [0,L] is an analytic integration fixture,
    # independent of the physical slab-mode checks above.
    class ConstantField:
        def theta(self,z):
            return np.ones_like(z,dtype=complex)/np.sqrt(.7)
    length,gamma = .7,3.
    kernel = HomogeneousGreen(1.,1.,np.sqrt(10.))
    exact = (length/gamma**2+np.expm1(-gamma*length)/gamma**3)/length
    value = green_overlap(ConstantField(),(0.,length),(0.,length),kernel,10)
    assert value == pytest.approx(exact,rel=2e-14)
    # Public coefficient diagnostics keep layer axes and the nu sign.
    c = build(dual=True)
    assert c.mu.shape == (4,2,4)
    assert c.varsigma.shape == (4,2,2,4)
    m,n = c.aux_orders[0]
    indices = {tuple(o):i for i,o in enumerate(c.xi_orders)}
    source = c.xi[:,indices[(m-1,n)]]
    np.testing.assert_allclose(c.nu[0,:,0],-c.J*source,atol=2e-15)
