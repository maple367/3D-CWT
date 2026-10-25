"""New-API acceptance: no legacy import and no oracle-only acceptance."""
import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cwt3d.greens import CutoffError, HomogeneousGreen


def ode_relative_residual(g, z, source=0.0, h=1e-4):
    value = g(z, source)
    second = (g(z + h, source) - 2 * value + g(z - h, source)) / h**2
    potential = (g.k0**2 * g.epsilon - g.k_parallel**2) * value
    return np.max(np.abs(second + potential) / np.abs(potential))


def test_lossless_radiation_independent_exponential():
    g = HomogeneousGreen(k0=2, epsilon=2.25)
    z = np.array([-2., -.3, 0., .4, 2.])
    expected = -1j / 6 * np.exp(-3j * np.abs(z))
    assert_allclose(g(z, 0), expected, rtol=2e-15)
    assert g.gamma == 3j
    assert g.kz == 3
    assert np.iscomplexobj(g(0, 0))


def test_task2_actual_new_api_repairs_high_order_physical_regression():
    # Task 2 fixture: k0=1, beta0=3, order=(2,0) => k_parallel=6.
    g = HomogeneousGreen(1, 1 + 0j, 6)
    z = np.array([-1., -.2, .2, 1.])
    assert_allclose(g(z, 0), np.exp(-np.sqrt(35) * abs(z)) / (2 * np.sqrt(35)))
    assert ode_relative_residual(g, z) < 1e-7
    assert abs(g(1, 0)) < abs(g(.2, 0))
    assert_allclose(g.derivative(0, 0, side='right') - g.derivative(0, 0, side='left'), -1)


@pytest.mark.parametrize('epsilon,k_parallel', [(2.25, 0), (1, 6), (2.25-.4j, .3), (-2-.1j, 1)])
def test_source_free_ode_continuity_jump_and_analytic_derivative(epsilon, k_parallel):
    g = HomogeneousGreen(1.3, epsilon, k_parallel)
    z = np.array([-1.1, -.3, .2, .9])
    assert ode_relative_residual(g, z) < 2e-7
    h = 1e-6
    assert_allclose((g(z+h, 0)-g(z-h, 0))/(2*h), g.derivative(z, 0), atol=1e-10, rtol=1e-9)
    assert_allclose(g(-h, 0), g(h, 0), atol=0, rtol=0)
    assert_allclose(g(h, 0), g(0, 0), atol=6e-7, rtol=0)
    assert_allclose(g.derivative(0, 0, side='left'), .5)
    assert_allclose(g.derivative(0, 0, side='right'), -.5)
    assert abs(g.derivative(h, 0)-g.derivative(-h, 0)+1) < 7e-6
    # A separate source jump estimate from G itself, not its derivative method.
    slope_jump = (g(h, 0)-g(0, 0))/h - (g(0, 0)-g(-h, 0))/h
    assert abs(slope_jump + 1) < 4e-6
    with pytest.raises(ValueError, match='source'):
        g.derivative([-.1, 0, .1], 0)


@pytest.mark.parametrize('epsilon,k_parallel', [(4-.2j, 0), (4-.2j, 5), (-3-.2j, 0), (-3, 0), (4, 0)])
def test_passive_branch_outgoing_phase_and_decay(epsilon, k_parallel):
    g = HomogeneousGreen(2, epsilon, k_parallel)
    # Independent kz branch oracle, not gamma or the implementation formula.
    kz = np.sqrt(complex(4*epsilon-k_parallel**2))
    if kz.real < 0 or kz.imag > 0:
        kz = -kz
    z = np.array([0., .2, 1.])
    assert_allclose(g(z, 0), -1j/(2*kz)*np.exp(-1j*kz*z), rtol=3e-15)
    assert g.gamma.real >= 0 and g.gamma.imag >= 0
    assert g.kz.real >= 0 and g.kz.imag <= 0
    # Positive real outgoing wave flux for this time convention: -Im(G* G').
    assert -np.imag(np.conj(g(.4, 0))*g.derivative(.4, 0)) >= 0
    assert abs(g(2, 0)) <= abs(g(0, 0))


def test_symmetry_translation_scaling_and_same_parallel_order():
    g = HomogeneousGreen(2, 3-.2j, 4)
    z = np.array([-.4, .2, .7])[:, None]
    s = np.array([-.6, .3])[None, :]
    assert_allclose(g(z, s), g(s, z))
    assert_allclose(g(z+7, s+7), g(z, s), atol=2e-15)
    scale = 3.7
    scaled = HomogeneousGreen(2/scale, 3-.2j, 4/scale)
    assert_allclose(scaled(z*scale, s*scale), scale*g(z, s))
    assert_allclose(scaled.derivative(z*scale, s*scale), g.derivative(z, s))
    zero = HomogeneousGreen(2, 3)
    assert_allclose(zero(z, s), HomogeneousGreen(2, 3, 0)(z, s))
    assert_allclose(HomogeneousGreen(2, 3, np.hypot(3, 4))(z, s), HomogeneousGreen(2, 3, 5)(z, s))


@pytest.mark.filterwarnings('ignore:Setting the shape on a NumPy array:DeprecationWarning')
def test_broadcasting_shape_and_result_mutation_do_not_pollute():
    g = HomogeneousGreen(1, 1, 2)
    z = np.arange(3.)[:, None]
    s = np.array([.1, .3])[None, :]
    original_z, original_s = z.copy(), s.copy()
    expected = g(z, s).copy()
    assert expected.shape == (3, 2)
    assert g.derivative(z, s).shape == (3, 2)
    exposed = g(z, s)
    exposed.shape = (6,)
    exposed[:] = 123j
    assert_allclose(g(z, s), expected, rtol=0, atol=0)
    assert_allclose(z, original_z)
    assert_allclose(s, original_s)
    assert g([], 0).shape == (0,)
    assert np.shape(g(0, 1)) == ()
    assert g(np.ones((2, 1, 3)), np.ones((4, 1))).shape == (2, 4, 3)
    with pytest.raises(ValueError):
        g(np.ones(2), np.ones(3))


@pytest.mark.parametrize('name', ['k0', 'epsilon', 'k_parallel', 'gamma', 'kz'])
def test_configuration_is_immutable(name):
    g = HomogeneousGreen(1, 2-.1j, 3)
    before = g([0, .3], 0)
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        setattr(g, name, 7)
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        delattr(g, name)
    assert_allclose(g([0, .3], 0), before, rtol=0, atol=0)


def test_scalar_array_configuration_has_no_mutable_alias():
    k0, epsilon, parallel = np.array(1.), np.array(2-.1j), np.array(3.)
    g = HomogeneousGreen(k0, epsilon, parallel)
    expected = g(.2, 0)
    k0[...] = 7
    epsilon[...] = 9
    parallel[...] = 0
    assert g(.2, 0) == expected
    assert isinstance(g.k0, float) and isinstance(g.epsilon, complex)


@pytest.mark.parametrize('kwargs', [
    {'k0': 0}, {'k0': -1}, {'k0': True}, {'k0': np.nan}, {'k0': np.inf},
    {'k0': 1+0j}, {'k0': [1]}, {'epsilon': np.nan}, {'epsilon': 1+np.inf*1j},
    {'epsilon': True}, {'epsilon': [1]}, {'epsilon': '1'}, {'epsilon': 1+.01j},
    {'epsilon': 1+1e-300j}, {'k_parallel': -1}, {'k_parallel': False},
    {'k_parallel': np.inf}, {'k_parallel': np.nan}, {'k_parallel': 1j},
    {'k_parallel': [0]}, {'k0': 1e308}, {'epsilon': 1e308, 'k0': 2},
])
def test_invalid_configuration(kwargs):
    config = dict(k0=1, epsilon=2, k_parallel=0)
    config.update(kwargs)
    with pytest.raises(ValueError):
        HomogeneousGreen(**config)


@pytest.mark.parametrize('bad', [True, [True, False], [0, True], 1j, [1+0j], np.nan, np.inf, '1', None, object()])
def test_invalid_coordinates(bad):
    g = HomogeneousGreen(1, 2)
    for method in (g, g.derivative):
        with pytest.raises(ValueError):
            method(bad, 0)
        with pytest.raises(ValueError):
            method(0, bad)


def test_cutoff_is_explicit_but_small_nonzero_gamma_is_not_clamped():
    with pytest.raises(CutoffError, match='cutoff'):
        HomogeneousGreen(1, 1, 1)
    with pytest.raises(CutoffError):
        HomogeneousGreen(1, 0, 0)
    eps = np.nextafter(1., 0.)
    g = HomogeneousGreen(1, eps, 1)
    expected_gamma = np.sqrt(1-eps)
    assert_allclose(g(0, 0), 1/(2*expected_gamma), rtol=1e-15)
    assert abs(g(0, 0)) > 1e7
    assert_allclose(HomogeneousGreen(1, -1e-28)(0, 0), 5e13)


def test_thick_distance_is_stable_under_strict_seterr():
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for epsilon in (1, 1-.2j):
            g = HomogeneousGreen(1, epsilon, 6)
            assert_allclose(g([-1e6, 1e6], 0), [0j, 0j], atol=0)
            assert_allclose(g.derivative([-1e6, 1e6], 0), [0j, 0j], atol=0)


def test_derivative_side_validation_and_nonsource_invariance():
    g = HomogeneousGreen(1, 2)
    for side in ('left', 'right'):
        assert_allclose(g.derivative([-.2, .2], 0, side=side), g.derivative([-.2, .2], 0))
    for side in ('average', 0, [], True):
        with pytest.raises(ValueError):
            g.derivative(1, 0, side=side)
