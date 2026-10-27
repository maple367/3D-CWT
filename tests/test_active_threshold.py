"""Frozen gain spectra checked against independent dense generalized oracles."""
import numpy as np
import pytest
from scipy.linalg import eigvals
from scipy.optimize import linear_sum_assignment
from cwt3d.active.cavity import ActiveCavity
from cwt3d.active.threshold import solve_active_modes, uniform_threshold_gain
from cwt3d.finite.assembly import assemble_pencil
from cwt3d.finite.grid import StaggeredGrid


class Material:
    def gain(self,N,U): return .07*N
    def index_change(self,N): return .003*N


def test_uniform_threshold_formula():
    assert uniform_threshold_gain(.03,confinement=.2,internal_loss_um_inverse=.01) == pytest.approx(.35)
    assert uniform_threshold_gain(0,confinement=1) == 0


@pytest.mark.parametrize('kw',[dict(alpha=-1),dict(alpha=np.nan),dict(alpha=True),
    dict(alpha=[1]),dict(alpha=1,confinement=0),dict(alpha=1,confinement=1.1),
    dict(alpha=1,internal_loss_um_inverse=-1)])
def test_threshold_validation(kw):
    options=dict(confinement=.2)
    options.update(kw)
    with pytest.raises(ValueError): uniform_threshold_gain(**options)


def test_uniform_eigenshifts_and_patterned_dense_oracle():
    r=np.random.default_rng(17)
    C=.1*(r.normal(size=(4,4))+1j*r.normal(size=(4,4)))
    p=assemble_pencil(StaggeredGrid(2,2,10,8),C)
    a=ActiveCavity(p,group_index=3,k0_um_inverse=6,active_thickness_um=.1,
                   confinement=.2,internal_loss_um_inverse=.01)
    passive=eigvals(p.A.toarray(),p.B.toarray())
    for N in (0.,2.,np.arange(4).reshape(2,2)):
        g=.07*N; dn=.003*N
        shift=np.broadcast_to(-6*.2*dn-.5j*(.2*g-.01),(2,2))
        expected=eigvals(p.A.toarray()+np.diag(np.tile(shift.ravel(),4))@p.B.toarray(),p.B.toarray())
        A,B=a.frozen_pencil(N,Material())
        actual=eigvals(A.toarray(),B.toarray())
        i,j=linear_sum_assignment(abs(actual[:,None]-expected))
        np.testing.assert_allclose(actual[i],expected[j],atol=2e-12)
        if np.ndim(N)==0:
            i,j=linear_sum_assignment(abs(actual[:,None]-(passive+shift[0,0])))
            np.testing.assert_allclose(actual[i],(passive+shift[0,0])[j],atol=2e-12)
        result=solve_active_modes(a,N,Material(),sigma=.17+.21j,k=3)
        assert len(result.eigenvalues)==3
        assert np.max(result.residuals)<1e-8
        for z in result.eigenvalues:
            assert np.min(abs(expected-z))<1e-8


def test_uniform_threshold_cancels_mode_specific_loss():
    # This is a mode-specific fixed-gain identity, not a global mode search.
    alpha=.03; ai=.01; gamma=.2
    gain=uniform_threshold_gain(alpha,confinement=gamma,internal_loss_um_inverse=ai)
    assert (alpha-.5*(gamma*gain-ai)) == pytest.approx(0,abs=1e-16)
