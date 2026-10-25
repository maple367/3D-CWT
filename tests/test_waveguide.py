"""Independent open-boundary TE acceptance tests."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from cwt3d.waveguide import TEWaveguide, NoGuidedModeError


def slab_roots(k0, ec, es, width):
    # Independent even/odd half-core equations in transverse phase u.
    v = k0 * width / 2 * np.sqrt(ec-es)
    roots = []
    for n in range(int(2*v/np.pi)+1):
        lo, hi = n*np.pi/2, min((n+1)*np.pi/2, v)
        if hi <= lo:
            continue
        def f(u):
            w = np.sqrt(max(0., v*v-u*u))
            return u*np.tan(u)-w if n % 2 == 0 else -u/np.tan(u)-w
        a, b = lo+1e-10, hi-1e-10
        if f(a)*f(b) < 0:
            u = brentq(f, a, b, xtol=1e-14)
            roots.append(np.sqrt(k0*k0*ec-(2*u/width)**2))
    return np.array(roots)


def test_symmetric_slab_all_even_odd_modes():
    wg = TEWaveguide([2.25, 12.25, 2.25], [1.4], wavelength_um=1.55)
    modes = wg.solve_modes()
    expected = slab_roots(wg.k0, 12.25, 2.25, 1.4)
    assert len(modes) == len(expected) >= 4
    np.testing.assert_allclose([m.beta for m in modes], expected, rtol=2e-12)
    assert wg.solve_mode().beta == modes[0].beta
    assert wg.solve_mode(1).beta == modes[1].beta
    x = np.linspace(-.6, .6, 37)
    for i, m in enumerate(modes):
        np.testing.assert_allclose(m.theta(.7+x), (-1)**i*m.theta(.7-x), atol=2e-11)
        assert m.boundary_residual < 1e-10


def test_asymmetric_multilayer_continuity_norm_and_complex_broadcast():
    wg = TEWaveguide([1., 9., 4., 7., 2.25], [.4, .17, .3], k0=4.)
    for m in wg.solve_modes():
        for z in wg.interfaces_um:
            h = 1e-8
            assert abs(m.theta(z-h)-m.theta(z+h)) < 2e-6
            assert abs(m.derivative(z-h)-m.derivative(z+h)) < 2e-5
        norm = sum(quad(lambda z: abs(m.theta(z))**2, a, b, epsabs=1e-11)[0]
                   for a, b in zip(wg.interfaces_um[:-1], wg.interfaces_um[1:]))
        norm += abs(m.theta(0))**2/(2*np.sqrt(m.beta**2-wg.k0**2))
        norm += abs(m.theta(wg.interfaces_um[-1]))**2/(2*np.sqrt(m.beta**2-wg.k0**2*2.25))
        assert norm == pytest.approx(1., abs=2e-11)
        phase = 2-3j
        z = np.array([[-1., .1], [.65, 2.]])
        np.testing.assert_allclose(m.theta(z, amplitude=phase), phase*m.theta(z))
        np.testing.assert_allclose(m.derivative(z, amplitude=phase), phase*m.derivative(z))
        assert np.iscomplexobj(m.theta(z))
        assert np.isscalar(m.theta(.1))
        np.testing.assert_equal(m.eps_profile(z), [[1, 9], [7, 2.25]])
        g = np.sqrt(m.beta**2-16)
        assert m.theta(-2)/m.theta(-1) == pytest.approx(np.exp(-g))


def test_scale_covariance_and_fixed_period():
    a = TEWaveguide([1, 9, 2], [.6], k0=4, period_um=.7)
    b = TEWaveguide([1, 9, 2], [6.], k0=.4, period_um=7.)
    ma, mb = a.solve_mode(), b.solve_mode()
    assert ma.beta == pytest.approx(10*mb.beta, rel=1e-12)
    z = np.linspace(-1, 2, 101)
    np.testing.assert_allclose(ma.theta(z), np.sqrt(10)*mb.theta(10*z), atol=1e-11)
    np.testing.assert_allclose(ma.derivative(z), 10*np.sqrt(10)*mb.derivative(10*z), atol=1e-10)
    assert a.period_um == .7  # no implicit 2*pi/beta period update


def test_thick_evanescent_layer_and_equal_epsilon_subdivision():
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        a = TEWaveguide([1, 9, 1], [.6], k0=4).solve_mode()
        b = TEWaveguide([1, 9, 1, 1], [.6, 1e4], k0=4).solve_mode()
        assert a.beta == pytest.approx(b.beta, abs=2e-11)
        np.testing.assert_allclose(a.theta([-.1, .2, .7]), b.theta([-.1, .2, .7]), atol=2e-10)
        assert np.all(np.isfinite(b.theta([.7, 100., 9999., 10001.])))
        assert b.boundary_residual < 1e-10


def test_thick_internal_barrier_recovers_two_isolated_guides():
    left = TEWaveguide([1, 9, 1], [.6], k0=4).solve_modes()
    right = TEWaveguide([1, 7, 1], [.33], k0=4).solve_modes()
    expected = sorted([m.beta for m in left+right], reverse=True)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        modes = TEWaveguide([1, 9, 1, 7, 1], [.6, 1e4, .33], k0=4).solve_modes()
        np.testing.assert_allclose([m.beta for m in modes], expected, rtol=2e-12)
        for mode in modes:
            assert mode.boundary_residual < 1e-10
            assert np.all(np.isfinite(mode.theta([.2, 100., 9999., 10000.8])))


def test_equivalent_core_partition_and_amplitude_array():
    a = TEWaveguide([1, 9, 1], [.6], k0=4).solve_mode()
    b = TEWaveguide([1+0j, 9, 9, 1], [.21, .39], k0=4).solve_mode()
    z = np.linspace(-1, 2, 37)
    assert a.beta == pytest.approx(b.beta, abs=1e-12)
    np.testing.assert_allclose(a.theta(z), b.theta(z), atol=1e-12)
    phases = np.array([1, 1j, -2+3j])
    np.testing.assert_allclose(a.theta(.2, amplitude=phases), a.theta(.2)*phases)
    np.testing.assert_allclose(a.derivative(.2, amplitude=phases), a.derivative(.2)*phases)
    assert a.eps_profile(0.) == 9


def test_zero_kz_layer_limit():
    # Construct a middle epsilon equal to a known guided beta^2/k0^2;
    # inserting a constant-field layer at the even-mode center preserves beta.
    base = TEWaveguide([1, 9, 1], [.6], k0=4).solve_mode()
    ec = base.beta**2/16
    m = TEWaveguide([1, 9, ec, 9, 1], [.3, .2, .3], k0=4).solve_mode()
    assert m.beta == pytest.approx(base.beta, abs=1e-11)
    assert abs(m.derivative(.4)) < 1e-10
    assert np.isfinite(m.theta(.4))


@pytest.mark.parametrize('eps,d,kwargs', [
    ([1, 9, 1], [0], {'k0':4}), ([1, 9, 1], [np.inf], {'k0':4}),
    ([1, 9, 1], [], {'k0':4}), ([1, -9, 1], [.2], {'k0':4}),
    ([1, 9-.01j, 1], [.2], {'k0':4}), ([1, np.nan, 1], [.2], {'k0':4}),
    ([1, 9, 1], [.2], {}), ([1, 9, 1], [.2], {'k0':4, 'wavelength_um':1}),
    ([1, 9, 1], [.2], {'k0':0}), ([1, 9, 1], [.2], {'k0':4,'period_um':-1}),
])
def test_invalid_inputs(eps, d, kwargs):
    with pytest.raises((ValueError, TypeError)):
        TEWaveguide(eps, d, **kwargs)


def test_asymmetric_roots_against_independent_ode_shooting():
    from scipy.integrate import solve_ivp
    wg = TEWaveguide([1., 9., 4., 7., 2.25], [.4, .17, .3], k0=4.)
    for mode in wg.solve_modes():
        # Integrate the original ODE, with no production element/basis calls.
        state = np.array([1., np.sqrt(mode.beta**2-16)])
        for e, width in zip(wg.eps[1:-1], wg.thickness_um):
            def ode(z, y):
                return [y[1], (mode.beta**2-16*e)*y[0]]
            state = solve_ivp(ode, (0, width), state, rtol=2e-12, atol=2e-13).y[:, -1]
        residual = state[1]+np.sqrt(mode.beta**2-36)*state[0]
        assert abs(residual)/np.linalg.norm(state) < 2e-10
        z = np.array([.1, .5, .7])
        h = 1e-5
        second = (mode.theta(z+h)-2*mode.theta(z)+mode.theta(z-h))/h**2
        np.testing.assert_allclose(second, (mode.beta**2-16*mode.eps_profile(z))*mode.theta(z),
                                   rtol=2e-5, atol=2e-5)


def test_reject_unrepresentable_derived_wavenumber():
    with pytest.raises(ValueError, match='representable'):
        TEWaveguide([1, 9, 1], [.2], wavelength_um=1e-320)


@pytest.mark.parametrize('name,replacement', [
    ('k0', 5.), ('eps', (2., 10., 2.)), ('thickness_um', (.8,)),
    ('interfaces_um', (0., .8)), ('period_um', .9),
])
@pytest.mark.parametrize('operation', ['assign', 'delete'])
def test_waveguide_configuration_is_read_only(name, replacement, operation):
    guide = TEWaveguide([1, 9, 1], [.6], k0=4, period_um=.7)
    mode = guide.solve_mode()
    z = np.array([-.1, 0., .1, .3, .5, .6, .7])
    beta, residual = mode.beta, mode.boundary_residual
    field, derivative = mode.theta(z), mode.derivative(z)
    profile = mode.eps_profile(z)
    original = getattr(guide, name)
    with pytest.raises(AttributeError):
        if operation == 'assign':
            setattr(guide, name, replacement)
        else:
            delattr(guide, name)
    assert getattr(guide, name) == original
    assert mode.beta == beta
    assert mode.boundary_residual == residual
    np.testing.assert_array_equal(mode.theta(z), field)
    np.testing.assert_array_equal(mode.derivative(z), derivative)
    np.testing.assert_array_equal(mode.eps_profile(z), profile)
    again = guide.solve_mode()
    fresh = TEWaveguide([1, 9, 1], [.6], k0=4, period_um=.7).solve_mode()
    assert again.beta == fresh.beta == beta
    np.testing.assert_array_equal(again.theta(z), fresh.theta(z))
    np.testing.assert_array_equal(again.derivative(z), fresh.derivative(z))
    # A different configuration requires a new guide, not an in-place edit.
    changed = TEWaveguide([1, 9, 1], [.6], k0=5).solve_mode()
    assert changed.beta != beta
    np.testing.assert_array_equal(mode.theta(z), field)


@pytest.mark.parametrize('name', ['guide', 'beta', 'boundary_residual'])
@pytest.mark.parametrize('operation', ['assign', 'delete'])
def test_mode_public_state_is_read_only(name, operation):
    mode = TEWaveguide([1, 9, 1], [.6], k0=4).solve_mode()
    z = np.array([-.1, .1, .3, .7])
    original = getattr(mode, name)
    beta, residual = mode.beta, mode.boundary_residual
    field, derivative = mode.theta(z), mode.derivative(z)
    replacement = (TEWaveguide([1, 9, 1], [.6], k0=5)
                   if name == 'guide' else 0.)
    with pytest.raises(AttributeError):
        if operation == 'assign':
            setattr(mode, name, replacement)
        else:
            delattr(mode, name)
    assert getattr(mode, name) == original
    assert mode.beta == beta
    assert mode.boundary_residual == residual
    np.testing.assert_array_equal(mode.theta(z), field)
    np.testing.assert_array_equal(mode.derivative(z), derivative)


def test_waveguide_configuration_does_not_expose_mutable_aliases():
    eps, widths = np.array([1., 9., 1.]), np.array([.6])
    guide = TEWaveguide(eps, widths, k0=4)
    mode = guide.solve_mode()
    z = np.array([-.1, .1, .3, .7])
    field, derivative = mode.theta(z), mode.derivative(z)
    beta = mode.beta
    eps[:] = 2.
    widths[:] = .9
    for name in ('eps', 'thickness_um', 'interfaces_um'):
        values = getattr(guide, name)
        assert isinstance(values, tuple)
        with pytest.raises(TypeError):
            values[0] = 100.
    # Evaluation outputs must not alias either the configuration or mode state.
    guide.eps_profile(z)[:] = 100.
    mode.theta(z)[:] = 100.
    mode.derivative(z)[:] = 100.
    mode.eps_profile(z)[:] = 100.
    assert guide.eps == (1., 9., 1.)
    assert guide.thickness_um == (.6,)
    assert guide.interfaces_um == (0., .6)
    assert mode.guide is guide
    assert mode.beta == beta == guide.solve_mode().beta
    np.testing.assert_array_equal(mode.theta(z), field)
    np.testing.assert_array_equal(mode.derivative(z), derivative)


def test_no_guided_modes_and_index_validation():
    wg = TEWaveguide([4, 1, 4], [.5], k0=4)
    assert wg.solve_modes() == []
    with pytest.raises(NoGuidedModeError):
        wg.solve_mode()
    wg = TEWaveguide([1, 9, 1], [.2], k0=4)
    for index in (-1, 1, 1.5, True):
        with pytest.raises((ValueError, TypeError, IndexError)):
            wg.solve_mode(index)
