"""Synthetic algebraic checks; these numbers are not device material fits."""
import numpy as np
import pytest

from cwt3d.active.materials import CarrierMaterial


def test_exact_laws_and_signed_trials():
    m = CarrierMaterial(2., 3., gain_compression_um3=.5,
                        index_per_density_um3=-.2, recombination_a_ns=1.,
                        recombination_b_um3_ns=2., recombination_c_um6_ns=3.)
    n = np.array([[-1., 3., 5.]])
    np.testing.assert_allclose(m.gain(n, 2.), n-3)
    np.testing.assert_allclose(m.index_change(n), -.2*(n-3))
    np.testing.assert_allclose(m.recombination(n), n+2*n*n+3*n*n*n)
    assert m.gain(1., 0.) == -4.
    assert m.gain(3., 1.) == 0.


@pytest.mark.filterwarnings('ignore:Setting the (shape|dtype|strides):DeprecationWarning')
def test_snapshot_metadata_and_frozen_parameters():
    a = np.array([[1., 2.], [3., 4.]])
    m = CarrierMaterial(a, 0.)
    a[:] = 99
    for operation in ('shape', 'dtype', 'strides'):
        public = m.differential_gain_um2
        assert isinstance(public.base, bytes)
        with pytest.raises(ValueError):
            public.setflags(write=True)
        if operation == 'shape':
            public.shape = (4,)
        elif operation == 'dtype':
            public.dtype = np.uint8
        else:
            public.strides = (8, 16)
        np.testing.assert_array_equal(m.differential_gain_um2, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(m.gain(np.ones((2, 2)), 0), [[1, 2], [3, 4]])
    for name in ('differential_gain_um2', 'transparency_density_um3', '_parameters'):
        with pytest.raises((AttributeError, TypeError)):
            setattr(m, name, 2.)
        with pytest.raises((AttributeError, TypeError)):
            delattr(m, name)


@pytest.mark.parametrize('value', [True, [True], 1j, '2', np.nan, np.inf, [1, 2], [[0.]]])
def test_invalid_differential_gain(value):
    with pytest.raises(ValueError):
        CarrierMaterial(value, 0.)


@pytest.mark.parametrize('name', ['transparency_density_um3', 'gain_compression_um3',
                                  'recombination_a_ns', 'recombination_b_um3_ns',
                                  'recombination_c_um6_ns'])
def test_negative_physical_parameters(name):
    args = {'differential_gain_um2': 1., 'transparency_density_um3': 0., name: -1.}
    with pytest.raises(ValueError):
        CarrierMaterial(**args)


@pytest.mark.parametrize('bad', [True, 1j, np.nan, np.inf, '3'])
def test_invalid_evaluation(bad):
    m = CarrierMaterial(1., 0.)
    for method in (m.gain, m.index_change, m.recombination):
        with pytest.raises(ValueError):
            method(bad, 0) if method == m.gain else method(bad)
    with pytest.raises(ValueError):
        m.gain(1., bad)


def test_shapes_no_accidental_outer_broadcast():
    with pytest.raises(ValueError):
        CarrierMaterial(np.ones((2, 1)), np.ones((1, 3)))
    m = CarrierMaterial(np.ones((2, 3)), 1.)
    assert m.gain(2., 0.).shape == (2, 3)
    for n, u in [(np.ones((2, 1)), 0.), (np.ones(6), 0.),
                 (np.ones((2, 3)), np.ones((1, 3)))]:
        with pytest.raises(ValueError):
            m.gain(n, u)
    with pytest.raises(ValueError):
        m.gain(1., -1.)


@pytest.mark.parametrize('name', [
    'differential_gain_um2', 'transparency_density_um3', 'gain_compression_um3',
    'index_per_density_um3', 'recombination_a_ns', 'recombination_b_um3_ns',
    'recombination_c_um6_ns'])
def test_every_parameter_getter_is_independent_bytes_snapshot(name):
    source = np.arange(1., 7.).reshape(2, 3)
    args = dict(differential_gain_um2=1., transparency_density_um3=0.)
    args[name] = source
    m = CarrierMaterial(**args)
    source[:] = 100
    first, second = getattr(m, name), getattr(m, name)
    assert first is not second
    assert isinstance(first.base, bytes) and isinstance(second.base, bytes)
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0, 0] = 123
    np.testing.assert_array_equal(second, np.arange(1., 7.).reshape(2, 3))
    with pytest.raises((AttributeError, TypeError)):
        setattr(m, name, source)
    with pytest.raises((AttributeError, TypeError)):
        delattr(m, name)


def test_patterned_parameters_exact_laws():
    x = np.arange(1., 7.).reshape(2, 3)
    m = CarrierMaterial(x, x+1, gain_compression_um3=x/10,
                        index_per_density_um3=-x,
                        recombination_a_ns=x, recombination_b_um3_ns=x/2,
                        recombination_c_um6_ns=x/3)
    n, u = x-2, x/4
    np.testing.assert_allclose(m.gain(n, u), x*(-3)/(1+x*u/10))
    np.testing.assert_allclose(m.index_change(n), 3*x)
    np.testing.assert_allclose(m.recombination(n), x*n+x/2*n**2+x/3*n**3)


def test_overflow_fails_closed():
    with pytest.raises(ValueError):
        CarrierMaterial(1e308, 0.).gain(1e308, 0.)
    with pytest.raises(ValueError):
        CarrierMaterial(1., 0., gain_compression_um3=1e308).gain(1., 1e308)
    with pytest.raises(ValueError):
        CarrierMaterial(1., 0., recombination_c_um6_ns=1.).recombination(1e308)
    with pytest.raises(ValueError):
        CarrierMaterial(1., 0., index_per_density_um3=1e308).index_change(1e308)
