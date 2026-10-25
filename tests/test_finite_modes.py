"""Finite fields: independent face integrals, physical coupling and identity tracking."""
import numpy as np
import pytest
import warnings
from cwt3d.finite import StaggeredGrid, assemble_pencil
from cwt3d.finite.eigensolver import EigenResult, solve_finite
from cwt3d.finite.fields import reconstruct_field, reconstruct_mode, reconstruct_modes
from cwt3d.finite.tracking import track_modes


def pencil(nx=3, ny=2, gain=0.):
    rng = np.random.default_rng(120)
    c = rng.normal(size=(4,4))*.09 + 1j*rng.normal(size=(4,4))*.09
    c = c+c.conj().T + 1j*gain*np.eye(4)
    return assemble_pencil(StaggeredGrid(nx,ny,7.,5.),c)


def integrals(mode, c):
    g, f = mode.grid, mode.faces
    # Independent full face interpolation; no B, budget or center getter.
    u = np.array([(f.Rx[:,:-1]+f.Rx[:,1:])/2,
                  (f.Sx[:,:-1]+f.Sx[:,1:])/2,
                  (f.Ry[:-1,:]+f.Ry[1:,:])/2,
                  (f.Sy[:-1,:]+f.Sy[1:,:])/2]).reshape(4,-1)
    area = g.hx_um*g.hy_um
    p = area*np.sum(abs(u)**2)
    edge = g.hy_um*(np.sum(abs(f.Rx[:,-1])**2)+np.sum(abs(f.Sx[:,0])**2))
    edge += g.hx_um*(np.sum(abs(f.Ry[-1,:])**2)+np.sum(abs(f.Sy[0,:])**2))
    pc = 2*area*np.vdot(u,c@u).imag
    return p, edge, pc


@pytest.mark.parametrize('nx,ny',[(1,1),(1,4),(4,1),(3,2)])
@pytest.mark.parametrize('gain',[0.,-.4,.17])
def test_actual_modes_full_face_energy_closure(nx,ny,gain):
    p = pencil(nx,ny,gain)
    r = solve_finite(p,k=min(4,p.grid.n_dof-2),sigma=.07+.4j,tol=1e-12)
    modes = reconstruct_modes(p,r)
    assert len(modes) == len(r.eigenvalues)
    for i,m in enumerate(modes):
        power,edge,pc = integrals(m,p.C)
        assert power == pytest.approx(1,abs=4e-15)
        assert m.eigenvalue == r.eigenvalues[i]
        assert m.delta == m.eigenvalue.real and m.alpha == m.eigenvalue.imag
        assert 2*m.alpha*power == pytest.approx(edge+pc,abs=2e-10)
        assert m.budget['edge'] == pytest.approx(edge)
        assert m.budget['coupling'] == pytest.approx(pc,abs=1e-15)
        assert m.budget['balance_relative_error'] < 1e-9
        assert m.boundary_error == 0
        assert m.reconstruction_error < 2e-15
        assert m.residual < 1e-10
        np.testing.assert_allclose(p.grid.pack(m.faces),m.free_vector,atol=0)
        np.testing.assert_allclose(p.B@m.free_vector,m.center_vector,atol=2e-15)
        assert np.sum(m.intensity)*p.grid.hx_um*p.grid.hy_um == pytest.approx(1)
        if gain == 0:
            assert abs(pc) < 1e-14
        elif gain < 0:
            assert pc < 0  # signed gain must not be absolutized


def test_scale_phase_immutable_and_partial():
    p = pencil()
    r = solve_finite(p,k=3,sigma=.1+.4j)
    original = r.eigenvectors[:,0].copy()
    base = reconstruct_mode(p,r,0)
    for factor in (2j,-7+3j,1e200,1e-200):
        vector = original*factor
        m = reconstruct_field(p,vector,r.eigenvalues[0])
        vector[:] = 0
        np.testing.assert_allclose(m.free_vector,base.free_vector,atol=3e-15)
        assert m.budget['edge'] == pytest.approx(base.budget['edge'])
    for name in ('free_vector','center_vector','intensity','component_intensities'):
        a = getattr(base,name)
        before = a.copy()
        assert isinstance(a.base,bytes)
        with pytest.raises(ValueError): a.setflags(write=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            a.shape = (a.size,)
            a.strides = (0,)
        np.testing.assert_array_equal(getattr(base,name),before)
    with pytest.raises(TypeError): base.budget['edge'] = 0
    # Older CPython frozen+slots dataclasses raise this specific TypeError
    # for non-field properties; the invariant is rejection without mutation.
    alpha_before = base.alpha
    with pytest.raises((AttributeError, TypeError)) as frozen:
        base.alpha = 99
    if isinstance(frozen.value, TypeError):
        assert str(frozen.value) == 'super(type, obj): obj must be an instance or subtype of type'
    assert base.alpha == alpha_before
    partial = EigenResult(r.eigenvalues[:1],r.eigenvectors[:,:1],r.residuals[:1],
                          {'returned':1,'requested':3,'converged':False})
    assert len(reconstruct_modes(p,partial)) == 1
    no_phase = reconstruct_field(p,original*1j,r.eigenvalues[0],phase_fix=False)
    assert abs(np.vdot(no_phase.center_vector,base.center_vector))*p.grid.hx_um*p.grid.hy_um == pytest.approx(1)


@pytest.mark.parametrize('bad',[np.zeros(24),np.ones(23),np.ones((24,1)),np.full(24,np.nan),np.full(24,np.inf)])
def test_invalid_vectors(bad):
    with pytest.raises(ValueError): reconstruct_field(pencil(),bad,.3j)


@pytest.mark.parametrize('index',[-1,True,1.5,3])
def test_invalid_index(index):
    p = pencil(); r = solve_finite(p,k=3)
    with pytest.raises((ValueError,IndexError,TypeError)): reconstruct_mode(p,r,index)


def test_real_physical_coupling_and_parameter_tracking():
    from test_coupling import build
    c = build(dual=True)
    p = assemble_pencil(StaggeredGrid(4,3,15.,12.),c)
    r = solve_finite(p,k=6,sigma=.025+.12j,tol=1e-12)
    modes = reconstruct_modes(p,r,coupling=c)
    for m in modes:
        power,edge,pc = integrals(m,c.C)
        assert m.budget['radiation_scope'] == 'all radiative coupling (auxiliary orders evanescent)'
        assert abs(m.budget['C1D']) < 2e-14
        assert abs(m.budget['C2D']) < 2e-14
        assert m.budget['Crad'] >= -1e-14
        assert 2*m.alpha*power == pytest.approx(edge+m.budget['Crad'],abs=2e-10)
        assert pc == pytest.approx(m.budget['C1D']+m.budget['Crad']+m.budget['C2D'],abs=1e-14)
    c2 = build(dual=True,contrast=2.001)
    p2 = assemble_pencil(p.grid,c2)
    newer = reconstruct_modes(p2,solve_finite(p2,k=6,sigma=.025+.12j,tol=1e-12),coupling=c2)
    tracking = track_modes(modes,newer,min_overlap=.8)
    assert len(tracking.matched) == 6
    assert not any(tracking.ambiguous) and not any(tracking.ambiguous_new)
    assert min(tracking.scores[tracking.assignment >= 0]) > .99
    for i,j in tracking.matched:
        aligned = reconstruct_field(p2,tracking.aligned_vectors[i],newer[j].eigenvalue,phase_fix=False)
        assert aligned.residual < 1e-10
        assert aligned.eigenvalue == newer[j].eigenvalue
        np.testing.assert_allclose(abs(aligned.free_vector),abs(newer[j].free_vector),atol=2e-15)
    with pytest.raises(ValueError,match='coupling'): reconstruct_modes(p,r,coupling=c2)


def fixtures(values, vectors=None):
    p = pencil(2,1)
    if vectors is None: vectors = np.eye(p.grid.n_dof,dtype=complex)[:,:len(values)]
    return tuple(reconstruct_field(p,vectors[:,i],value,phase_fix=False) for i,value in enumerate(values))


def test_tracking_permutation_phase_crossing_missing_new():
    old = fixtures([-.1j,1-.2j,2-.3j])
    # Same known fields, eigenvalues cross; frequency sorting gives the wrong identity.
    new = fixtures([4-.2j,-3-.1j,7j],np.eye(8,dtype=complex)[:,[1,0,6]]*np.array([1j,-1,1]))
    t = track_modes(old,new,min_overlap=.9)
    np.testing.assert_array_equal(t.assignment,[1,0,-1])
    assert t.matched == ((0,1),(1,0))
    assert t.unmatched_old == (2,) and t.unmatched_new == (2,)
    for i,j in t.matched:
        v = t.aligned_vectors[i]
        overlap = np.vdot(old[i].center_vector,old[i].grid.hx_um*old[i].grid.hy_um*(pencil(2,1).B@v))
        assert overlap.real > 0 and abs(overlap.imag) < 1e-14
        np.testing.assert_allclose(abs(v),abs(new[j].free_vector))
    assert track_modes((),new).unmatched_new == (0,1,2)
    assert track_modes(old,()).unmatched_old == (0,1,2)
    with pytest.raises(ValueError,match='grid'):
        track_modes(old,(reconstruct_field(pencil(),np.ones(24),1j),))


def test_exact_degeneracy_unitary_rotation_subspace_not_lost():
    # Orthogonal cell-center fields in different components; individual overlap=.707.
    basis = np.eye(8,dtype=complex)[:,[0,2]]
    old = fixtures([.2+.3j,.2+.3j],basis)
    q = np.array([[1,1j],[1j,1]])/np.sqrt(2)
    new = fixtures([.2+.3j,.2+.3j],basis@q)
    t = track_modes(old,new,min_overlap=.95)
    assert t.matched == ()  # deliberately no strong individual identity
    assert t.unmatched_old == t.unmatched_new == ()
    assert t.ambiguous == (True,True)
    assert len(t.clusters) == 1
    np.testing.assert_allclose(t.clusters[0].principal_overlaps,[1,1],atol=1e-14)
    assert t.clusters[0].old_indices == (0,1)
    assert t.clusters[0].new_indices == (0,1)
    assert t.aligned_vectors == (None,None)


def test_small_power_relative_balance_not_hidden_by_unit_floor():
    # Physical length rescaling changes rates by 1e-12, not their relative closure.
    p = pencil(); r = solve_finite(p,k=2,sigma=.1+.4j)
    g = p.grid
    huge = assemble_pencil(StaggeredGrid(g.Nx,g.Ny,g.Lx_um*1e12,g.Ly_um*1e12),p.C/1e12)
    m = reconstruct_field(huge,r.eigenvectors[:,0],r.eigenvalues[0]/1e12 + 1e-14j)
    assert m.budget['balance_absolute_error'] < 1e-12
    assert m.budget['balance_relative_error'] > .001


@pytest.mark.parametrize('defect',['nonfinite','shape','sum'])
def test_reject_inconsistent_coupling_components(defect):
    from dataclasses import replace
    from test_coupling import build
    from cwt3d.coupling import _pack
    c = build()
    bad = c.C1D.copy()
    if defect == 'nonfinite': bad[0,0] = np.nan
    elif defect == 'shape': bad = bad[:2,:2]
    else: bad[0,0] += 1j
    arrays = tuple(entry for entry in c._arrays if entry[0] != 'C1D')+_pack(C1D=bad)
    corrupt = replace(c,_arrays=arrays)
    p = assemble_pencil(StaggeredGrid(2,1,7.,5.),c)
    with pytest.raises(ValueError,match='component'):
        reconstruct_field(p,np.ones(8),.2j,coupling=corrupt)


def test_propagating_aux_label_and_tracking_metadata():
    from dataclasses import replace
    from test_coupling import build
    # Controlled metadata branch; not a second propagating-aux physics oracle.
    c = replace(build(),all_aux_evanescent=False)
    p = assemble_pencil(StaggeredGrid(2,1,7.,5.),c)
    m = reconstruct_field(p,np.ones(8),.2j,coupling=c)
    assert m.budget['radiation_scope'] == 'zero-order only; auxiliary orders may radiate'
    old = fixtures([.2j,.3j]); new = fixtures([.3j,.2j],np.eye(8)[:,[1,0]])
    t = track_modes(old,new,min_overlap=.9)
    for name in ('assignment','scores','overlap'):
        a = getattr(t,name); before = a.copy()
        assert isinstance(a.base,bytes)
        with pytest.raises(ValueError): a.setflags(write=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            a.shape = (a.size,); a.strides = (0,)
        np.testing.assert_array_equal(getattr(t,name),before)
    for kw in ({'min_overlap':1.01},{'min_overlap':np.nan},{'degeneracy_atol':-1},{'degeneracy_rtol':True}):
        with pytest.raises(ValueError): track_modes(old,new,**kw)


def test_near_degeneracy_reports_cluster_only_no_mixed_eigenvectors():
    basis = np.eye(8,dtype=complex)[:,[0,2]]
    old = fixtures([.2+.3j,.2+.300000001j],basis)
    new = fixtures([.2+.3j,.2+.300000001j],basis@np.array([[1,1],[1,-1]])/np.sqrt(2))
    t = track_modes(old,new,min_overlap=.9)
    assert len(t.clusters) == 1 and t.aligned_vectors == (None,None)
    assert t.ambiguous == (True,True)
    a = t.clusters[0].principal_overlaps
    assert isinstance(a.base,bytes)
    with pytest.raises(ValueError): a.setflags(write=True)
    # Rank-deficient candidate bases must not falsely certify a full subspace.
    deficient = fixtures([.2+.3j,.2+.3j],basis[:,[0,0]])
    assert track_modes(old,deficient,min_overlap=.9).clusters == ()


def test_hungarian_eligibility_before_assignment():
    from cwt3d.finite.tracking import _assign
    # Unconstrained sum maximization picks the diagonal then rejects one pair;
    # eligible maximum-cardinality assignment must keep BOTH cross matches.
    scores = np.array([[1.,.7],[.7,.6]])
    assert _assign(scores,scores>=.7) == ((0,1),(1,0))


@pytest.mark.parametrize('reverse,phase',[(False,1),(True,1j)])
def test_normal_pencil_assignment_tie_is_not_identity(reverse,phase):
    g = StaggeredGrid(1,1,1.,1.)
    c = np.diag(np.arange(4.))
    q = np.eye(4); q[:2,:2] = np.array([[1,1],[1,-1]])/np.sqrt(2)
    po,pn = assemble_pencil(g,c),assemble_pencil(g,q@c@q.T)
    old = tuple(reconstruct_field(po,np.eye(4)[:,i],i+2j) for i in range(3))
    new = tuple(reconstruct_field(pn,q[:,i]*phase,i+2j,phase_fix=False) for i in range(3))
    if reverse: new = new[::-1]
    assert max(m.residual for m in old+new) < 2e-16
    t = track_modes(old,new)
    assert t.matched == ((2,0 if reverse else 2),)
    assert t.ambiguous == (True,True,False)
    assert t.assignment_ambiguous_old == (True,True,False)
    assert t.assignment_ambiguous_new == ((False,True,True) if reverse else (True,True,False))
    assert t.unmatched_old == t.unmatched_new == ()
    assert t.aligned_vectors[:2] == (None,None)
    assert np.all(t.assignment[:2] == -1)


@pytest.mark.parametrize('scores,expected,amb_o,amb_n',[
    ([[.8,.8],[.8,.1]],((0,1),(1,0)),(False,False),(False,False)),
    ([[.9,.8],[.8,.9]],((0,0),(1,1)),(False,False),(False,False)),
    ([[.8,.8,.8]],(),(True,),(True,True,True)),
    ([[.8],[.8],[.8]],(),(True,True,True),(True,)),
    ([[1,0,0],[0,.8,.8],[0,.8,.8]],((0,0),),(False,True,True),(False,True,True)),
    ([[.8,.8],[.8,.8+1e-15]],(),(True,True),(True,True)),
    ([[.8,.8],[.8,.8+1e-10]],((0,0),(1,1)),(False,False),(False,False)),
    ([[.8,.8,.7]],(),(True,),(True,True,False)),
])
def test_global_assignment_ambiguity(scores,expected,amb_o,amb_n):
    from cwt3d.finite.tracking import _analyze_assignment
    scores = np.asarray(scores)
    pairs,ao,an = _analyze_assignment(scores,scores>=.7)
    assert pairs == expected
    assert ao == amb_o and an == amb_n


def test_unit_overlap_roundoff_single_and_subspace():
    from cwt3d.finite.tracking import _eligible
    eps = np.finfo(float).eps
    assert _eligible(np.array([1-eps]),1.).all()
    assert not _eligible(np.array([1-1e-10]),1.).any()
    rng = np.random.default_rng(42)
    old = fixtures([.2j],rng.normal(size=(8,1))+1j*rng.normal(size=(8,1)))
    t = track_modes(old,old,min_overlap=1)
    assert t.overlap[0,0] >= 1-32*eps  # measured as 0.9999999999999999 here
    assert t.matched == ((0,0),)
    basis = np.eye(8)[:,[0,2]]
    deg = fixtures([.2j,.2j],basis)
    rotated = fixtures([.2j,.2j],basis@np.array([[1,1],[1,-1]])/np.sqrt(2))
    t = track_modes(deg,rotated,min_overlap=1)
    assert len(t.clusters) == 1
    assert t.ambiguous == (True,True)
    assert t.assignment_ambiguous_old == (False,False)


def test_cluster_assignment_tie_not_certified():
    basis = np.eye(8)[:,[0,2]]
    old = fixtures([.2j,.2j],basis)
    new = fixtures([.3j,.3j,.4j,.4j],np.column_stack((basis,basis)))
    t = track_modes(old,new,min_overlap=1)
    assert t.clusters == ()
    assert t.assignment_ambiguous_old == (True,True)
    assert t.assignment_ambiguous_new == (True,True,True,True)
    assert t.unmatched_old == t.unmatched_new == ()


def test_cluster_assignment_preserves_forced_subspace():
    p = pencil(1,1)
    basis = np.eye(4)
    old = tuple(reconstruct_field(p,basis[:,i],(1+i//2)*1j) for i in range(4))
    new = tuple(reconstruct_field(p,basis[:,i],(3+j//2)*1j)
                for j,i in enumerate([0,1,0,1,2,3]))
    t = track_modes(old,new,min_overlap=1)
    assert len(t.clusters) == 1
    assert t.clusters[0].old_indices == (2,3)
    assert t.clusters[0].new_indices == (4,5)
    assert t.assignment_ambiguous_old == (True,True,False,False)
    assert t.assignment_ambiguous_new == (True,True,True,True,False,False)
    assert t.ambiguous == (True,True,True,True)
    assert t.unmatched_old == t.unmatched_new == ()


@pytest.mark.parametrize('transpose',[False,True])
def test_rectangular_public_assignment_uncertainty(transpose):
    g = StaggeredGrid(1,1,1.,1.)
    p = assemble_pencil(g,np.diag(np.arange(4.)))
    old = (reconstruct_field(p,np.eye(4)[:,0],2j),)
    q = np.eye(4); q[:2,:2] = np.array([[1,1],[1,-1]])/np.sqrt(2)
    pn = assemble_pencil(g,q@p.C@q.T)
    new = tuple(reconstruct_field(pn,q[:,i],i+2j) for i in range(2))
    if transpose: old,new = new,old
    t = track_modes(old,new)
    assert t.matched == ()
    assert all(t.assignment_ambiguous_old) and all(t.assignment_ambiguous_new)
    assert t.unmatched_old == t.unmatched_new == ()
    assert all(v is None for v in t.aligned_vectors)


def test_assignment_analysis_against_independent_small_enumeration():
    # Exhaustive enumeration is a TEST oracle only, never a production path.
    from math import fsum
    from cwt3d.finite.tracking import _analyze_assignment
    rng = np.random.default_rng(721)
    for n,m in ((1,4),(4,1),(2,3),(3,2),(3,3),(4,4)):
        for _ in range(12):
            scores = rng.integers(0,5,size=(n,m))/4
            eligible = scores>=.5
            options = []
            def visit(i,pairs,used):
                if i == n:
                    options.append(frozenset(pairs)); return
                visit(i+1,pairs,used)
                for j in range(m):
                    if j not in used and eligible[i,j]:
                        visit(i+1,pairs+[(i,j)],used|{j})
            visit(0,[],set())
            k = max(map(len,options))
            options = [p for p in options if len(p) == k]
            value = max(fsum(scores[i,j] for i,j in p) for p in options)
            opts = [p for p in options if fsum(scores[i,j] for i,j in p) == value]
            forced = set.intersection(*(set(p) for p in opts))
            ao = tuple(len({next((j for a,j in p if a == i),-1) for p in opts})>1 for i in range(n))
            an = tuple(len({next((i for i,b in p if b == j),-1) for p in opts})>1 for j in range(m))
            pairs,actual_o,actual_n = _analyze_assignment(scores,eligible)
            assert set(pairs) == forced
            assert (actual_o,actual_n) == (ao,an)


def test_diagnostics_recompute_residual_instead_of_trusting_result():
    p = pencil()
    fake = EigenResult(np.array([.2+.3j]),np.ones((24,1),complex),np.zeros(1),{'returned':1})
    m = reconstruct_mode(p,fake,0)
    assert m.residual > .01
    assert m.budget['balance_relative_error'] > .01
    with pytest.raises(ValueError): reconstruct_field(p,np.ones(24),np.nan)
    with pytest.raises(ValueError): reconstruct_field(p,np.ones(24),.3j,phase_fix='yes')
    wrong = EigenResult(np.array([.2j]),np.ones((23,1),complex),np.zeros(1),{})
    with pytest.raises(ValueError): reconstruct_modes(p,wrong)
