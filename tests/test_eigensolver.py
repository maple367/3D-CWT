"""Sparse non-Hermitian solver checked against independent dense QZ oracles."""
import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence

from cwt3d.finite import StaggeredGrid, assemble_pencil
from cwt3d.finite.eigensolver import (
    solve_generalized, solve_finite, FactorizationError, EigenSolveError,
)
import cwt3d.finite.eigensolver as solver


def synthetic(n=18):
    rng = np.random.default_rng(72)
    b = np.eye(n) + .12*(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    a = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    a += np.diag(np.linspace(-6,6,n))
    return sparse.csr_matrix(a), sparse.csr_matrix(b)


def check_oracle(a, b, result, k, sigma, atol=2e-8):
    reference = linalg.eig(a.toarray(), b.toarray(), right=False)
    reference = reference[np.isfinite(reference)]
    reference = reference[np.argsort(abs(reference-sigma))[:k]]
    rows, cols = linear_sum_assignment(abs(result.eigenvalues[:,None]-reference))
    np.testing.assert_allclose(result.eigenvalues[rows], reference[cols], atol=atol, rtol=atol)
    assert result.eigenvectors.shape == (a.shape[0],k)
    for i, value in enumerate(result.eigenvalues):
        v = result.eigenvectors[:,i]
        av, bv = a@v, b@v
        denominator = (sparse.linalg.norm(a)+abs(value)*sparse.linalg.norm(b))*np.linalg.norm(v)
        residual = np.linalg.norm(av-value*bv)/denominator
        assert result.residuals[i] == pytest.approx(residual, abs=2e-16)
        assert residual < 1e-9
    np.testing.assert_array_equal(result.delta, result.eigenvalues.real)
    np.testing.assert_array_equal(result.alpha, result.eigenvalues.imag)
    np.testing.assert_array_equal(result.power_loss, 2*result.alpha)
    return float(np.max(abs(result.eigenvalues[rows]-reference[cols])))


def test_general_complex_nonhermitian_b_and_repeat():
    a,b = synthetic()
    assert np.linalg.norm((b-b.conj().T).toarray()) > .1
    r = solve_generalized(a,b,k=4,sigma=.2+.3j,seed=13,tol=1e-12)
    check_oracle(a,b,r,4,.2+.3j)
    assert r.converged and r.status == 'converged'
    assert r.requested == r.returned == 4
    np.testing.assert_array_equal(r.eigenvalues, solve_generalized(a,b,k=4,sigma=.2+.3j,seed=13,tol=1e-12).eigenvalues)
    assert r.lu_nnz > 0 and r.linear_solves > 0
    assert r.lu_seconds >= 0 and r.iteration_seconds >= 0
    assert np.any(r.alpha < 0)  # no abs/clipping of signed threshold compensation


def test_finite_complex_c_and_minimum_grid():
    rng = np.random.default_rng(9)
    c = .1*(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))
    for nx,ny in ((1,1),(3,2)):
        p = assemble_pencil(StaggeredGrid(nx,ny,7.,5.),c)
        r = solve_finite(p,sigma=.1+.4j,tol=1e-12)
        check_oracle(p.A,p.B,r,min(6,p.grid.n_dof-2),.1+.4j)


def test_one_lu_operator_only_no_double_shift(monkeypatch):
    a,b = synthetic()
    lu_calls, eig_calls = [], []
    original_lu, original_eigs = solver.splu, solver.eigs
    def lu(matrix):
        lu_calls.append(matrix.copy())
        assert sparse.isspmatrix_csc(matrix)
        return original_lu(matrix)
    def eig(operator, **kw):
        eig_calls.append(kw)
        assert isinstance(operator, solver.LinearOperator)
        assert 'sigma' not in kw and 'M' not in kw
        assert kw['which'] == 'LM'
        return original_eigs(operator,**kw)
    monkeypatch.setattr(solver,'splu',lu)
    monkeypatch.setattr(solver,'eigs',eig)
    def forbidden(*args,**kwargs):
        raise AssertionError('dense conversion/inversion forbidden in production')
    monkeypatch.setattr(sparse.csr_matrix,'toarray',forbidden)
    monkeypatch.setattr(sparse.csc_matrix,'toarray',forbidden)
    monkeypatch.setattr(np.linalg,'inv',forbidden)
    monkeypatch.setattr(linalg,'eig',forbidden)
    r = solve_generalized(a,b,k=3,sigma=.25j)
    assert len(lu_calls) == len(eig_calls) == 1
    assert sparse.linalg.norm(lu_calls[0]-(a-.25j*b)) < 1e-14
    assert r.linear_solves > 3


def test_inputs_snapshots_and_global_state():
    import warnings
    a,b = synthetic()
    aa,bb = a.copy(),b.copy()
    v = np.linspace(1,2,a.shape[0])+1j*np.linspace(-1,1,a.shape[0])
    before = v.copy()
    state, err = np.random.get_state(), np.geterr()
    r = solve_generalized(a,b,k=3,v0=v,seed=7)
    np.testing.assert_array_equal(v,before)
    assert sparse.linalg.norm(a-aa) == sparse.linalg.norm(b-bb) == 0
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1],after[1])
    assert err == np.geterr()
    for name in ('eigenvalues','eigenvectors','delta','alpha','power_loss','residuals'):
        array = getattr(r,name)
        expected = array.copy()
        assert isinstance(array.base,bytes)
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            array.shape = (array.size,)
            array.dtype = np.uint8
            array.strides = (0,)
        np.testing.assert_array_equal(getattr(r,name),expected)
    with pytest.raises((AttributeError,TypeError)):
        r.status = 'wrong'
    with pytest.raises(TypeError):
        r.metadata['status'] = 'wrong'
    a.data[:] = 0
    b.data[:] = 0
    check_oracle(aa,bb,r,3,0)
    later = solve_generalized(aa,bb,k=3,sigma=-.5+.2j)
    check_oracle(aa,bb,later,3,-.5+.2j)
    check_oracle(aa,bb,r,3,0)


@pytest.mark.parametrize('kw',[
    {'k':0},{'k':17},{'k':True},{'k':1.5},{'sigma':np.inf},{'sigma':1j*np.nan},
    {'tol':-1},{'tol':np.nan},{'tol':True},{'maxiter':0},{'maxiter':1.5},
    {'ncv':3,'k':3},{'ncv':4,'k':3},{'ncv':19},{'ncv':True},
    {'seed':-1},{'seed':True},{'seed':1.5},{'residual_tol':0},{'residual_tol':np.inf},
    {'v0':np.zeros(18)},{'v0':np.ones((18,1))},{'v0':np.full(18,np.nan)},
])
def test_invalid_parameters(kw):
    with pytest.raises((ValueError,TypeError)):
        solve_generalized(*synthetic(),**kw)


def test_invalid_matrices_and_integer_sparse_support():
    a,b = synthetic()
    for x,y in ((a.toarray(),b),(a,b[:,:-1]),(a[:,:-1],b),
                (sparse.eye(2),sparse.eye(2))):
        with pytest.raises((ValueError,TypeError)):
            solve_generalized(x,y)
    a.data[0] = np.nan
    with pytest.raises(ValueError,match='finite'):
        solve_generalized(a,b)
    a = sparse.diags(np.arange(1,9),dtype=int,format='csr')
    r = solve_generalized(a,sparse.eye(8,dtype=int),sigma=2.3,k=2)
    check_oracle(a,sparse.eye(8),r,2,2.3)


def test_singular_shift_and_close_nonzero_shift():
    a,b = sparse.diags([1.,2.,3.,4.,5.,6.]),sparse.eye(6)
    with pytest.raises(FactorizationError,match='shift'):
        solve_generalized(a,b,sigma=2,k=2)
    r = solve_generalized(a,b,sigma=2+1e-10j,k=1,tol=1e-12)
    check_oracle(a,b,r,1,2+1e-10j)


def test_partial_exception_and_explicit_opt_in(monkeypatch):
    a,b = synthetic()
    values,vectors = linalg.eig(a.toarray(),b.toarray())
    index = np.argmin(abs(values))
    def partial(*args,**kwargs):
        raise ArpackNoConvergence('controlled branch only',1/values[index:index+1],vectors[:,index:index+1])
    monkeypatch.setattr(solver,'eigs',partial)
    with pytest.raises(EigenSolveError) as error:
        solve_generalized(a,b,k=3)
    r = error.value.partial
    assert not r.converged and not r.arpack_converged
    assert r.returned == 1 and r.requested == 3 and r.status == 'partial'
    assert max(r.residuals) < 1e-12
    opt = solve_generalized(a,b,k=3,allow_partial=True)
    np.testing.assert_array_equal(opt.eigenvalues,r.eigenvalues)


def test_empty_partial_always_fails(monkeypatch):
    def partial(*args,**kwargs):
        raise ArpackNoConvergence('controlled empty',np.empty(0,complex),np.empty((18,0),complex))
    monkeypatch.setattr(solver,'eigs',partial)
    with pytest.raises(EigenSolveError) as error:
        solve_generalized(*synthetic(),k=3,allow_partial=True)
    assert error.value.partial.returned == 0
    assert error.value.partial.status == 'failed'


def test_invalid_transforms_and_false_arpack_success(monkeypatch):
    a,b = synthetic()
    values,vectors = linalg.eig(a.toarray(),b.toarray())
    index = np.argmin(abs(values))
    def fake(*args,**kwargs):
        return np.array([1/values[index],0,np.inf,1j]),np.column_stack([
            vectors[:,index],np.ones(18),np.ones(18),np.ones(18)])
    monkeypatch.setattr(solver,'eigs',fake)
    r = solve_generalized(a,b,k=4,allow_partial=True)
    assert r.arpack_converged and not r.converged
    assert r.returned == 1 and r.discarded_transform == 2 and r.discarded_residual == 1
    with pytest.raises(EigenSolveError):
        solve_generalized(a,b,k=4)


def test_singular_b_finite_subset_and_diagnostic(monkeypatch):
    a = sparse.diags(np.arange(1.,11.))
    b = sparse.diags([1.]*8+[0.,0.])
    r = solve_generalized(a,b,k=2,sigma=2.2)
    check_oracle(a,b,r,2,2.2)
    # Zero mu represents infinite eigenvalues; never convert to physical zero.
    monkeypatch.setattr(solver,'eigs',lambda *args,**kw:(np.zeros(2,complex),np.eye(10,2,dtype=complex)))
    with pytest.raises(EigenSolveError) as error:
        solve_generalized(a,b,k=2,allow_partial=True)
    assert error.value.partial.discarded_transform == 2


def test_real_arpack_iteration_limit():
    a,b = synthetic(90)
    with pytest.raises(EigenSolveError) as error:
        solve_generalized(a,b,k=5,ncv=8,maxiter=1,tol=1e-14)
    assert not error.value.partial.arpack_converged
    assert not error.value.partial.converged


def test_extreme_finite_v0_scale_is_normalized_without_global_seterr():
    a,b = synthetic()
    for magnitude in (1.7e308,1e-320):
        v = np.full(18,complex(magnitude,magnitude))
        with np.errstate(all='raise'):
            r = solve_generalized(a,b,k=2,v0=v,tol=1e-12)
        check_oracle(a,b,r,2,0)


def test_zero_eigenvalue_is_not_a_fake_infinite_mode():
    a = sparse.diags([0.,1.,2.,3.,4.,5.])
    b = sparse.eye(6)
    r = solve_generalized(a,b,k=1,sigma=.1,tol=1e-12)
    assert r.converged and abs(r.eigenvalues[0]) < 1e-12
    assert r.residuals[0] < 1e-12


@pytest.mark.parametrize('allow_partial', [False, True])
@pytest.mark.parametrize('nonhermitian', [False, True])
def test_all_zero_finite_spectrum_real_arpack(allow_partial, nonhermitian):
    a = sparse.csr_matrix((6,6))
    b = sparse.eye(6,format='csr')
    if nonhermitian:
        b = b + sparse.diags([.3j]*5,1,shape=(6,6),format='csr')
    r = solve_generalized(a,b,sigma=.25,k=2,allow_partial=allow_partial)
    assert r.converged and r.returned == 2
    np.testing.assert_array_equal(r.eigenvalues,0)
    for i, value in enumerate(r.eigenvalues):
        v = r.eigenvectors[:,i]
        assert np.linalg.norm(v) > 0 and np.linalg.norm(b@v) > 0
        assert np.linalg.norm(a@v-value*(b@v)) == r.residuals[i] == 0


@pytest.mark.parametrize('b', [sparse.diags([1.,1.,1.,1.,1.,0.]),
                              sparse.csr_matrix((6,6))])
def test_zero_pencil_undefined_spectrum_fails(b):
    with pytest.raises(FactorizationError):
        solve_generalized(sparse.csr_matrix((6,6)),b,sigma=.25,k=2)


def test_true_tiny_nonzero_spectrum_is_not_snapped():
    b = sparse.eye(6,format='csr')
    # Tiny in absolute units, resolved relative to its own pencil scale.
    a = sparse.diags(np.arange(1.,7.)*1e-24,format='csr')
    r = solve_generalized(a,b,sigma=.25e-24,k=2,tol=1e-12)
    np.testing.assert_allclose(r.eigenvalues,[1e-24,2e-24],rtol=1e-12,atol=0)
    assert np.all(r.eigenvalues != 0)
    check_oracle(a,b,r,2,.25e-24,atol=1e-35)


@pytest.mark.parametrize('zero_a', [False, True])
def test_large_shift_does_not_admit_bad_pairs(monkeypatch, zero_a):
    a = sparse.csr_matrix((6,6)) if zero_a else sparse.diags(np.arange(1.,7.))
    b = sparse.eye(6)
    # A naive shift-scaled denominator would accept this lambda=1e3 pair.
    sigma = 1e12
    wrong_value = 1e3
    monkeypatch.setattr(solver,'eigs',lambda *args,**kw:(
        np.array([1/(wrong_value-sigma)]), np.ones((6,1),complex)))
    for allow_partial in (False, True):
        with pytest.raises(EigenSolveError) as error:
            solve_generalized(a,b,sigma=sigma,k=1,allow_partial=allow_partial)
        assert error.value.partial.returned == 0
        assert error.value.partial.discarded_residual == 1


@pytest.mark.parametrize('bad_vector', [np.zeros(6), np.full(6,np.nan)])
def test_zero_spectrum_still_rejects_invalid_vectors(monkeypatch, bad_vector):
    monkeypatch.setattr(solver,'eigs',lambda *args,**kw:(
        np.array([-4.+0j]),bad_vector[:,None]))
    with pytest.raises(EigenSolveError):
        solve_generalized(sparse.csr_matrix((6,6)),sparse.eye(6),
                          sigma=.25,k=1,allow_partial=True)


def test_zero_spectrum_partial_certificate_filter(monkeypatch):
    # One consistent candidate survives, while a false transform is rejected.
    monkeypatch.setattr(solver,'eigs',lambda *args,**kw:(
        np.array([-4.+0j,-2.+0j]),np.eye(6,2,dtype=complex)))
    a, b = sparse.csr_matrix((6,6)), sparse.eye(6)
    with pytest.raises(EigenSolveError) as error:
        solve_generalized(a,b,sigma=.25,k=2)
    r = solve_generalized(a,b,sigma=.25,k=2,allow_partial=True)
    assert r.returned == error.value.partial.returned == 1
    assert not r.converged and r.discarded_residual == 1
    np.testing.assert_array_equal(r.eigenvalues,[0j])
    assert r.residuals[0] == 0
    with pytest.raises(FactorizationError):
        solve_generalized(a,b,sigma=0,k=2)


def test_real_residual_gate_overrides_arpack_success():
    with pytest.raises(EigenSolveError) as error:
        solve_generalized(*synthetic(),k=3,tol=1e-12,residual_tol=1e-30)
    assert error.value.partial.arpack_converged
    assert not error.value.partial.converged
    assert error.value.partial.discarded_residual == 3


def _diagonal_reference(value, vector):
    # Independent standard-scale oracle: no production norm/residual helpers.
    from math import hypot
    v = np.asarray(vector, dtype=complex)
    numerator = hypot(*(abs((np.arange(1., 7.)-value)*v)))
    return numerator / ((np.sqrt(91.)+abs(value)*np.sqrt(6.))*hypot(*abs(v)))


@pytest.mark.parametrize('scale', [1., 1e-170, 1e170])
@pytest.mark.parametrize('allow_partial', [False, True])
def test_common_pencil_scale_rejects_wrong_pair(monkeypatch, scale, allow_partial):
    monkeypatch.setattr(solver, 'eigs', lambda *args, **kw: (
        np.array([1/1.5]), np.eye(6, 1, dtype=complex)))
    assert _diagonal_reference(1.5, np.eye(6)[:, 0]) > 1e-8
    with pytest.raises(EigenSolveError) as error:
        solve_generalized(scale*sparse.diags(np.arange(1., 7.)),
                          scale*sparse.eye(6), k=1, allow_partial=allow_partial)
    assert error.value.partial.discarded_residual == 1
    assert error.value.partial.returned == 0


@pytest.mark.parametrize('scale', [1., 1e-170, 1e170])
@pytest.mark.parametrize('amplitude', [1., 1e-170, 1e170])
def test_common_pencil_scale_correct_pair_residual(monkeypatch, scale, amplitude):
    v = np.array([1., 2e-10j, 0, 0, 0, 0], dtype=complex)
    value = 1.+1e-10
    monkeypatch.setattr(solver, 'eigs', lambda *args, **kw: (
        np.array([1/value]), (amplitude*v)[:, None]))
    r = solve_generalized(scale*sparse.diags(np.arange(1., 7.)),
                          scale*sparse.eye(6), k=1)
    assert r.converged
    assert r.residuals[0] > 0
    assert r.residuals[0] == pytest.approx(_diagonal_reference(r.eigenvalues[0], v), rel=2e-6)


@pytest.mark.parametrize('scale', [1., 1e-170, 1e170])
def test_common_pencil_scale_real_strict_gate(monkeypatch, scale):
    actual_eigs, actual_residual = solver.eigs, solver._residual
    observed, reported = [], []
    def record_residual(*args):
        result = actual_residual(*args)
        reported.append(result)
        return result
    def record(*args, **kwargs):
        mu, vectors = actual_eigs(*args, **kwargs)
        observed.extend(_diagonal_reference(1/m, v) for m, v in zip(mu, vectors.T))
        return mu, vectors
    monkeypatch.setattr(solver, 'eigs', record)
    monkeypatch.setattr(solver, '_residual', record_residual)
    try:
        r = solve_generalized(scale*sparse.diags(np.arange(1., 7.)),
                              scale*sparse.eye(6), k=2, residual_tol=1e-30)
    except EigenSolveError as error:
        r = error.partial
    # No assumption that every real ARPACK pair must have nonzero residual.
    # Independent evaluation can differ at roundoff because cancellation is
    # reordered by common scaling; it must not turn O(eps) into exact zero.
    np.testing.assert_allclose(reported, observed, rtol=.3, atol=1e-30)
    assert r.discarded_residual == sum(x > 1e-30 for x in observed)
    assert r.returned == sum(x <= 1e-30 for x in observed)
    for value, v, residual in zip(r.eigenvalues, r.eigenvectors.T, r.residuals):
        assert residual == pytest.approx(_diagonal_reference(value, v), rel=.3, abs=1e-30)


@pytest.mark.parametrize('scale', [1e-170, 1e170])
def test_common_pencil_scale_preserves_tiny_nonzero_spectrum(scale):
    a = scale*sparse.diags(np.arange(1., 7.)*1e-24)
    b = scale*sparse.eye(6)
    r = solve_generalized(a, b, sigma=.25e-24, k=2, tol=1e-12)
    np.testing.assert_allclose(r.eigenvalues, [1e-24, 2e-24], rtol=1e-12, atol=0)
    assert np.all(r.eigenvalues != 0)


@pytest.mark.parametrize('scale', [1e-170, 1e170])
def test_common_pencil_scale_zero_certificate_and_partial(monkeypatch, scale):
    a, b = sparse.csr_matrix((6, 6)), scale*sparse.eye(6, format='csr')
    b += scale*sparse.diags([.3j]*5, 1, shape=(6, 6), format='csr')
    r = solve_generalized(a, b, sigma=.25, k=2)
    np.testing.assert_array_equal(r.eigenvalues, 0)
    np.testing.assert_array_equal(r.residuals, 0)
    def partial(*args, **kwargs):
        raise ArpackNoConvergence('scaled mixed candidates',
                                 np.array([1., 1/1.5]), np.eye(6, 2, dtype=complex))
    monkeypatch.setattr(solver, 'eigs', partial)
    a, b = scale*sparse.diags(np.arange(1., 7.)), scale*sparse.eye(6)
    with pytest.raises(EigenSolveError) as error:
        solve_generalized(a, b, k=2)
    r = solve_generalized(a, b, k=2, allow_partial=True)
    assert r.returned == error.value.partial.returned == 1
    assert r.status == 'partial' and not r.arpack_converged
    assert r.discarded_residual == 1
    np.testing.assert_array_equal(r.eigenvalues, [1.])


def test_residual_scaling_exhausted_range_fails_closed(monkeypatch):
    # A common normalization cannot retain both extremes: refuse the pencil,
    # rather than drop A's nonzero coefficients and invoke the A=0 special case.
    def forbidden(*args, **kwargs):
        raise AssertionError('unsafe residual pencil must fail before ARPACK')
    monkeypatch.setattr(solver, 'eigs', forbidden)
    with pytest.raises(ValueError, match='numeric range exhausted'):
        solve_generalized(1e-200*sparse.eye(6), 1e200*sparse.eye(6), sigma=1., k=1)


@pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo', 'dia', 'bsr'])
@pytest.mark.parametrize('phase', [1., 1j, 1.+1j])
@pytest.mark.parametrize('matrix_phase', [1., 1j, 1.+1j])
def test_sparse_residual_product_underflow_rejected(monkeypatch, fmt, phase, matrix_phase):
    # True ratio is |v[0]|/||v|| ~= 1e-150, although Av is 1e-350.
    v = np.array([phase*1e-150, 1, 0, 0, 0, 0], dtype=complex)
    a = sparse.diags([matrix_phase*1e-200, 0, 0, 0, 0, 0]).asformat(fmt)
    b = sparse.eye(6).asformat(fmt)
    monkeypatch.setattr(solver, 'eigs', lambda *args, **kw: (
        np.array([1.+0j]), v[:, None]))
    before = np.geterr()
    with np.errstate(all='raise'):
        for partial in (False, True):
            with pytest.raises(EigenSolveError) as error:
                solve_generalized(a, b, sigma=-1, k=1, residual_tol=1e-200,
                                  allow_partial=partial)
            r = error.value.partial
            assert r.returned == 0 and r.discarded_residual == 1
            assert r.discarded_numeric_range == 1
            assert 'numeric range' in str(error.value)
    assert np.geterr() == before


@pytest.mark.parametrize('coefficient,first', [(1e-200, 0.), (1e-100, 1e-150),
                                              (1e-200, 1.)])
@pytest.mark.parametrize('phase', [1., 1j, 1.+1j])
def test_sparse_residual_safe_products_and_exact_zero(monkeypatch, coefficient, first, phase):
    v = np.array([first*phase, 1, 0, 0, 0, 0], dtype=complex)
    a = sparse.diags([coefficient, 0, 0, 0, 0, 0])
    monkeypatch.setattr(solver, 'eigs', lambda *args, **kw: (
        np.array([1.+0j]), v[:, None]))
    with np.errstate(all='raise'):
        r = solve_generalized(a, sparse.eye(6), sigma=-1, k=1, residual_tol=2.)
    from math import hypot
    expected = abs(first*phase)/hypot(abs(first*phase), 1.)
    assert r.residuals[0] == pytest.approx(expected, rel=1e-14, abs=0.)
    assert r.discarded_numeric_range == 0


def test_residual_complex_scalar_product_range():
    # Matvecs are safe; only lambda*Bv contains a nonzero 1e-350 term.
    a = sparse.csr_matrix((3, 3), dtype=complex)
    b = sparse.eye(3, format='csr', dtype=complex)
    with pytest.raises(ValueError, match='numeric range'):
        solver._residual(a, b, 0., np.sqrt(3.), 1e-200j,
                         np.array([1e-150j, 1., 0.]))


def test_product_range_bound_is_conservative_and_zeros_are_exact():
    with np.errstate(all='raise'):
        solver._check_products(np.array([0., 1e-300j]), np.array([1e-300j, 0.]))
        solver._check_products(2.**-500, 2.**-522)  # exactly normal tiny
        with pytest.raises(ValueError, match='numeric range'):
            solver._check_products(2.**-500, 2.**-523)  # subnormal
        # Normal product, but rejected by the conservative mantissa bound.
        with pytest.raises(ValueError, match='numeric range'):
            solver._check_products(1.5*2.**-500, 1.5*2.**-523)


def test_real_optical_chain(capsys):
    from time import perf_counter
    from cwt3d.geometry import CircleHole
    from cwt3d.waveguide import TEWaveguide
    from cwt3d.coupling import build_coupling
    start = perf_counter()
    mode = TEWaveguide([1.,9.,1.],[.32],k0=6.).solve_mode()
    period = 2*np.pi/mode.beta
    f = np.pi*.13**2
    hole = CircleHole(.13*period,(period,period),(.36*period,.47*period),
                      9+2*f,9-2*(1-f))
    coupling = build_coupling(mode,period,{1:hole},cutoff=1,quadrature_order=12)
    assert np.linalg.norm(coupling.C-coupling.C.conj().T) > 1e-5
    pencil = assemble_pencil(StaggeredGrid(3,2,20*period,16*period),coupling)
    # Dense reference is TEST ONLY; fixed target independent of production.
    sigma = .02+.3j
    result = solve_finite(pencil,k=4,sigma=sigma,tol=1e-12,seed=21)
    oracle_error = check_oracle(pencil.A,pencil.B,result,4,sigma,atol=2e-7)
    fields = pencil.grid.unpack(result.eigenvectors[:,0])
    centers = pencil.grid.interpolate(fields)
    center_vector = np.concatenate([getattr(centers,key).ravel() for key in ('Rx','Sx','Ry','Sy')])
    np.testing.assert_allclose(center_vector,pencil.B@result.eigenvectors[:,0],atol=1e-15)
    np.testing.assert_array_equal(fields.Rx[:,0],0)
    np.testing.assert_array_equal(fields.Sx[:,-1],0)
    np.testing.assert_array_equal(fields.Ry[0,:],0)
    np.testing.assert_array_equal(fields.Sy[-1,:],0)
    print({'n':pencil.grid.n_dof,'period_um':period,'sigma':sigma,
           'k':result.k,'tol':result.tolerance,'residual_tol':result.residual_tolerance,
           'seed':result.seed,'ncv':result.ncv,'maxiter':result.maxiter,
           'oracle_error':oracle_error,
           'max_residual':max(result.residuals),'eigenvalues':result.eigenvalues.tolist(),
           'lu_seconds':result.lu_seconds,'iteration_seconds':result.iteration_seconds,
           'lu_nnz':result.lu_nnz,'linear_solves':result.linear_solves,
           'chain_seconds':perf_counter()-start})
    with capsys.disabled():
        print(capsys.readouterr().out)
