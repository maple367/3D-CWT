"""Matched-accuracy gates; the reference remains independently selectable."""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from cwt3d.coupling import build_coupling
from cwt3d.greens import LayeredGreen, PoleError
from test_coupling import physical


@pytest.mark.parametrize('dual', [False, True])
@pytest.mark.parametrize('cutoff', [1, 3])
def test_coupling_backend_parity(dual, cutoff):
    m, a, h = physical(dual=dual)
    ref = build_coupling(m, a, h, cutoff=cutoff, quadrature_order=8)
    fast = build_coupling(m, a, h, cutoff=cutoff, quadrature_order=8, backend='prepared')
    for name, *_ in ref._arrays:
        np.testing.assert_allclose(getattr(fast, name), getattr(ref, name), rtol=3e-12, atol=3e-14)
    r, f = np.linalg.eigvals(ref.C), np.linalg.eigvals(fast.C)
    i, j = linear_sum_assignment(abs(r[:, None]-f))
    assert max(abs(r[i]-f[j])) < 3e-13
    assert fast.fourier_order == ref.fourier_order == 64


def test_backend_validation_and_extreme_period():
    m, a, h = physical()
    with pytest.raises(ValueError, match='backend'):
        build_coupling(m, a, h, backend='other')
    for p in (np.nextafter(0., 1.), np.finfo(float).max):
        with pytest.raises(ValueError, match='beta0'):
            build_coupling(m, p, {}, beta_rtol=2)


@pytest.mark.parametrize('args', [
    (1., [1., 4., 2.], [.8], .3),
    (1., [1-.1j, 4-.3j, 2-.2j], [.8], 1.4),
    (1., [1., 4., 1.], [.8], 2.),  # internal cutoff
    (1., [1., 4., 1.], [1000.], 3.),
    (1., [1., 2.], [], .3),
])
def test_green_broadcast_derivative_and_snapshot(args):
    from cwt3d.greens import PreparedLayeredGreen
    ref, fast = LayeredGreen(*args), PreparedLayeredGreen(*args)
    nodes = np.array(ref._nodes)
    z = np.r_[-.3, nodes, nodes[-1]+.2, nodes[-1]*.37]
    s = z.copy()
    np.testing.assert_allclose(fast(z[:, None], s), ref(z[:, None], s), rtol=2e-12, atol=2e-13)
    for side in ('left', 'right'):
        np.testing.assert_allclose(fast.derivative(z[:, None], s, side=side),
                                   ref.derivative(z[:, None], s, side=side), rtol=3e-12, atol=3e-13)
    with pytest.raises(ValueError, match='side'):
        fast.derivative(z, s)
    assert np.isscalar(fast(.1, .2))
    assert fast(np.empty((0, 1)), s).shape == (0, len(s))


def test_prepared_pole_check():
    from cwt3d.greens import PreparedLayeredGreen
    # Analytic symmetric slab guided-mode pole.
    from cwt3d.waveguide import TEWaveguide
    m = TEWaveguide([1., 4., 1.], [.8], k0=1.).solve_mode()
    with pytest.raises(PoleError):
        PreparedLayeredGreen.from_structure(m.guide, m.beta)
    with pytest.raises(PoleError):
        LayeredGreen.from_structure(m.guide, m.beta)(.1, .2)


@pytest.mark.parametrize('gate', [
    'test_physical_ode_interfaces_source_jump_flux',
    'test_internal_cutoff_linear_limit',
    'test_thick_barrier_and_distant_points_strict',
    'test_true_guided_pole_and_nonzero_near_pole',
    'test_scaling_translation_and_structure_snapshot',
    'test_errors_leave_object_and_seterr_intact',
    'test_random_passive_and_negative_materials_independent_oracle',
    'test_element_dirichlet_resonance_is_not_global_pole',
    'test_lossy_thick_layer_and_strict_intermediate_underflow',
    'test_exterior_cutoff_and_resource_limits_are_explicit',
    'test_broadcast_metadata_and_profile_are_isolated',
])
def test_prepared_existing_independent_physical_gates(gate, monkeypatch):
    # Re-exercise the independent physical oracle suite, not just parity against
    # the implementation sharing shape/local helpers. Only select no-arg gates.
    import test_greens_layered as gates
    from cwt3d.greens import PreparedLayeredGreen
    monkeypatch.setattr(gates, 'LayeredGreen', PreparedLayeredGreen)
    getattr(gates, gate)()


def test_factorization_once_and_batch_rhs(monkeypatch):
    from cwt3d.greens import PreparedLayeredGreen
    import scipy.linalg as la
    calls = {'svd':0, 'lu':0, 'solve':0}
    originals = (np.linalg.svd, la.lu_factor, la.lu_solve)
    def count(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapped
    monkeypatch.setattr(np.linalg, 'svd', count('svd', originals[0]))
    monkeypatch.setattr(la, 'lu_factor', count('lu', originals[1]))
    monkeypatch.setattr(la, 'lu_solve', count('solve', originals[2]))
    g = PreparedLayeredGreen(1.,[1.,4.,1.],[.8],.4)
    g(np.linspace(-.2,1.,20)[:,None],np.linspace(-.1,.9,30))
    g.derivative(.2,.3)
    assert calls == {'svd':1,'lu':1,'solve':2}
