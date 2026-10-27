"""Independent synthetic limits; these fixtures are not a physical device."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eig

from cwt3d.active.dynamics import simulate, _jacobian_sparsity
from cwt3d.active.results import ActiveState, ActiveResult
from cwt3d.finite.assembly import assemble_pencil
from cwt3d.finite.grid import StaggeredGrid
from cwt3d.active.materials import CarrierMaterial


class SyntheticCavity:
    def __init__(self, nx=2, ny=2, gain=0.):
        self.grid = StaggeredGrid(nx, ny, 2., 2.)
        self.pencil = assemble_pencil(self.grid, np.eye(4)*(.2+.1j))
        self.n_dof = 4*nx*ny
        self.A, self.B = self.pencil.A, self.pencil.B
        self.vg_um_ns = 1.
        self.active_thickness_um = 1.
        self.confinement = 1.
        self.gain = gain
        self.radiation_scope = 'synthetic; not device radiation'

    def photon_density(self, v):
        return np.sum(abs(self.B.dot(v).reshape(4,self.grid.Ny,self.grid.Nx))**2, axis=0)

    def photon_number(self, v):
        return float(self.photon_density(v).sum()*self.grid.hx_um*self.grid.hy_um)

    def material_rate(self, n, u, material):
        return np.full(n.size, self.gain/2, dtype=complex)

    def power_budget(self, v, n, material):
        return {'total_photons': self.photon_number(v)}


class Material(CarrierMaterial):
    """Constant-gain synthetic limit; deliberately not a device material."""
    def __init__(self, recombination=0., gain=0.):
        super().__init__(1.,0.)
        object.__setattr__(self,'r',recombination)
        object.__setattr__(self,'g',gain)
    def gain(self,n,u):
        return np.full_like(n,self.g)
    def recombination(self,n):
        return self.r*n


def test_passive_exact_generalized_mode():
    c = SyntheticCavity()
    w,v = eig(c.A.toarray(), c.B.toarray())
    initial = ActiveState(v[:,0]*.1, 2.)
    t = np.linspace(0,.2,7)
    out = simulate(c, Material(), initial, times_ns=t, rtol=1e-9, field_atol=1e-12)
    expected = initial.field_vector[None,:]*np.exp(1j*c.vg_um_ns*w[0]*t[:,None])
    np.testing.assert_allclose(out.field_vectors,expected,rtol=2e-7,atol=2e-10)


def test_zero_field_injection_recombination():
    c = SyntheticCavity()
    from scipy.constants import elementary_charge
    j = 3.*elementary_charge*1e9
    t = np.linspace(0,1.,9)
    out = simulate(c,Material(2.),ActiveState(np.zeros(c.n_dof),4.),times_ns=t,
                   current_density_A_um2=j,rtol=1e-9,carrier_atol=1e-10)
    np.testing.assert_allclose(out.carrier_density_um3[:,0,0],1.5+2.5*np.exp(-2*t),rtol=1e-7)
    assert np.count_nonzero(out.field_vectors)==0


def test_gain_factor_two_and_time_refinement():
    c = SyntheticCavity(gain=.7)
    t = np.linspace(0,.3,6)
    v = np.arange(1,c.n_dof+1)*.001
    a = simulate(c,Material(),ActiveState(v,2.),times_ns=t,rtol=1e-9,field_atol=1e-13)
    c.gain = 0.
    b = simulate(c,Material(),ActiveState(v,2.),times_ns=t,rtol=1e-10,field_atol=1e-14,max_step_ns=.005)
    np.testing.assert_allclose(a.photon_number/b.photon_number,np.exp(.7*t),rtol=3e-7)


@pytest.mark.filterwarnings('ignore:Setting the (shape|dtype) on a NumPy array:DeprecationWarning')
def test_state_result_snapshots():
    v = np.ones(4,dtype=complex); n = np.ones((1,1)); md={'nested':{'a':np.arange(3)}}
    state = ActiveState(v,n); v[:]=7; n[:]=8
    assert state.field_vector[0]==1 and state.carrier_density_um3[0,0]==1
    view=state.field_vector; view.shape=(2,2)
    assert state.field_vector.shape==(4,)
    with pytest.raises(ValueError): state.field_vector[0]=3
    out=ActiveResult([0,1],np.ones((2,4)),np.ones((2,1,1)),[1,1],metadata=md)
    md['nested']['a'][0]=99
    assert out.metadata['nested']['a'][0]==0
    with pytest.raises(TypeError): out.metadata['x']=1
    array=out.metadata['nested']['a']; array.shape=(1,3)
    assert out.metadata['nested']['a'].shape==(3,)
    for name in ('times_ns','field_vectors','carrier_density_um3','photon_number'):
        original=getattr(out,name).copy()
        header=getattr(out,name)
        assert isinstance(header.base,bytes)
        with pytest.raises(ValueError): header.setflags(write=True)
        header.dtype=np.uint8
        np.testing.assert_array_equal(getattr(out,name),original)
    with pytest.raises((AttributeError,TypeError)):
        out._fields=None


@pytest.mark.parametrize('kwargs', [dict(times_ns=[0,0]),dict(times_ns=[-1,1]),dict(times_ns=[0,np.nan]),
    dict(times_ns=[0,1],current_density_A_um2=-1),dict(times_ns=[0,1],rtol=0),
    dict(times_ns=[0,1],max_output_bytes=1),dict(times_ns=[0,1],method='RK45')])
def test_bad_inputs(kwargs):
    c=SyntheticCavity()
    with pytest.raises((ValueError,TypeError)):
        simulate(c,Material(),ActiveState(np.zeros(c.n_dof),1.),**kwargs)


@pytest.mark.parametrize('nx,ny',[(3,2),(1,4),(4,1),(1,1)])
def test_sparse_support_includes_nonlocal_mass_inverse(nx,ny):
    c=SyntheticCavity(nx,ny)
    from cwt3d.active.carriers import diffusion_operator
    diffusion=diffusion_operator(c.grid,.3)
    p=_jacobian_sparsity(c.grid,c.A,c.B,diffusion)
    n=nx*ny; q=4*n
    # Compare full finite differences of a nonlinear synthetic coupled RHS.
    from scipy.sparse.linalg import splu
    lu=splu(c.B.tocsc())
    def fun(y):
        v=y[:q]+1j*y[q:2*q]; N=y[2*q:]; b=c.B@v
        U=np.sum(abs(b.reshape(4,n))**2,axis=0)
        m=N/(1+U)+1j*N
        dv=lu.solve(1j*c.A@v+np.tile(m,4)*b)
        dn=-N-U*N+diffusion@N
        return np.r_[dv.real,dv.imag,dn]
    y=np.linspace(.1,.9,9*n); f=fun(y)
    dense=p.toarray()
    for j in range(y.size):
        yp=y.copy(); yp[j]+=1e-6
        nonzero=abs((fun(yp)-f)/1e-6)>1e-7
        assert np.all(dense[nonzero,j])


@pytest.mark.parametrize('method',['BDF','Radau'])
def test_rotating_frame_passive_mode_and_invariant_observables(method):
    c=SyntheticCavity()
    w,v=eig(c.A.toarray(),c.B.toarray())
    seed=ActiveState(v[:,0]*.01,2.)
    t=np.linspace(0,.4,9); reference=.63
    plain=simulate(c,Material(),seed,times_ns=t,method=method,rtol=1e-9,field_atol=1e-13)
    rotated=simulate(c,Material(),seed,times_ns=t,method=method,rtol=1e-9,
                     field_atol=1e-13,reference_detuning_um_inverse=reference)
    exact=seed.field_vector[None,:]*np.exp(1j*c.vg_um_ns*(w[0]-reference)*t[:,None])
    np.testing.assert_allclose(rotated.field_vectors,exact,rtol=4e-7,atol=1e-10)
    np.testing.assert_allclose(plain.photon_number,rotated.photon_number,rtol=4e-7)
    np.testing.assert_allclose(plain.carrier_density_um3,rotated.carrier_density_um3)
    assert rotated.metadata['reference_detuning_um_inverse']==reference


@pytest.mark.parametrize('reference',[np.nan,np.inf,1j,True,[1.], '1'])
def test_reference_requires_strict_finite_real(reference):
    c=SyntheticCavity()
    with pytest.raises(ValueError):
        simulate(c,Material(),ActiveState(np.zeros(c.n_dof),1.),times_ns=[0,1],
                 reference_detuning_um_inverse=reference)


def test_diffusion_conserves_no_field_carriers():
    c=SyntheticCavity(3,2)
    N=np.array([[1.,3.,6.],[8.,2.,4.]])
    out=simulate(c,Material(),ActiveState(np.zeros(c.n_dof),N),
                 times_ns=np.linspace(0,1,9),diffusion_um2_ns=.7,
                 rtol=1e-9,carrier_atol=1e-10)
    np.testing.assert_allclose(out.carrier_density_um3.sum(axis=(1,2)),N.sum(),rtol=1e-12)
    assert np.var(out.carrier_density_um3[-1]) < np.var(N)


def test_reciprocal_stimulated_photon_carrier_balance():
    c=SyntheticCavity(gain=.6)
    c.A=sparse.csr_matrix(c.A.shape,dtype=complex)  # closed synthetic no-transport limit
    seed=ActiveState(np.arange(1,c.n_dof+1)*.01,4.)
    t=np.linspace(0,1,11)
    out=simulate(c,Material(gain=.6),seed,times_ns=t,rtol=1e-11,
                 carrier_atol=1e-12,field_atol=1e-14,max_step_ns=.01)
    np.testing.assert_allclose(out.photon_number,out.photon_number[0]*np.exp(.6*t),rtol=1e-7)
    carriers=out.carrier_density_um3.sum(axis=(1,2))*c.grid.hx_um*c.grid.hy_um
    np.testing.assert_allclose(carriers+out.photon_number,carriers[0]+out.photon_number[0],rtol=1e-10)


def test_solver_failure_and_negative_samples_are_not_hidden(monkeypatch):
    import cwt3d.active.dynamics as dynamics
    c=SyntheticCavity(); seed=ActiveState(np.zeros(c.n_dof),1.)
    def fake(rhs,span,y0,**kwargs):
        y=np.repeat(y0[:,None],2,axis=1)
        y[-1,-1]=-5e-5
        return SimpleNamespace(success=True,y=y,t=np.array([0.,1.]),message='test',nfev=1,njev=1,nlu=1)
    monkeypatch.setattr(dynamics,'solve_ivp',fake)
    out=simulate(c,Material(),seed,times_ns=[0,1],carrier_atol=1e-3)
    assert out.carrier_density_um3[-1,-1,-1]==-5e-5
    with pytest.raises(RuntimeError,match='below -carrier_atol'):
        simulate(c,Material(),seed,times_ns=[0,1],carrier_atol=1e-6)
    def failure(*args,**kwargs):
        return SimpleNamespace(success=False,message='deliberate failure')
    monkeypatch.setattr(dynamics,'solve_ivp',failure)
    with pytest.raises(RuntimeError,match='deliberate failure'):
        simulate(c,Material(),seed,times_ns=[0,1])


def test_callable_current_is_validated_at_each_evaluation():
    c=SyntheticCavity()
    def invalid_later(t):
        return 0. if t==0. else np.nan
    with pytest.raises(ValueError,match='finite'):
        simulate(c,Material(),ActiveState(np.zeros(c.n_dof),1.),times_ns=[0,1],
                 current_density_A_um2=invalid_later)


@pytest.mark.parametrize('field,carrier',[(np.ones((2,2)),1.),([np.inf],1.),([1],-1.),([1],[1]),([1],1j)])
def test_invalid_initial_snapshot(field,carrier):
    with pytest.raises(ValueError):
        ActiveState(field,carrier)


def test_real_cavity_nonlinear_maps_time_refinement():
    from cwt3d.active.cavity import ActiveCavity
    grid=StaggeredGrid(3,2,2.,2.)
    # Group index chosen only to make this numerical fixture vg=1 um/ns.
    from scipy.constants import c as light_speed
    cavity=ActiveCavity(assemble_pencil(grid,np.eye(4)*(.2+.1j)),
                        group_index=light_speed*1e-3,k0_um_inverse=6.,
                        active_thickness_um=.2,confinement=.4)
    material=CarrierMaterial(np.full((2,3),.1),np.full((2,3),1.),
                             gain_compression_um3=.2,index_per_density_um3=.02,
                             recombination_a_ns=.01)
    seed=ActiveState(np.linspace(.01,.12,grid.n_dof),np.full((2,3),3.))
    times=np.linspace(0,.2,7)
    coarse=simulate(cavity,material,seed,times_ns=times,diffusion_um2_ns=.03,
                    rtol=1e-5,carrier_atol=1e-8,field_atol=1e-10,
                    current_density_A_um2=np.zeros((2,3)))
    fine=simulate(cavity,material,seed,times_ns=times,diffusion_um2_ns=.03,
                  rtol=1e-9,carrier_atol=1e-11,field_atol=1e-13,max_step_ns=.005)
    np.testing.assert_allclose(coarse.field_vectors,fine.field_vectors,rtol=8e-5,atol=1e-7)
    np.testing.assert_allclose(coarse.carrier_density_um3,fine.carrier_density_um3,rtol=1e-6)
    np.testing.assert_allclose(fine.photon_number,[cavity.photon_number(v) for v in fine.field_vectors])
    rotated=simulate(cavity,material,seed,times_ns=times,diffusion_um2_ns=.03,
                     rtol=1e-9,carrier_atol=1e-11,field_atol=1e-13,max_step_ns=.005,
                     reference_detuning_um_inverse=.37)
    np.testing.assert_allclose(rotated.photon_number,fine.photon_number,rtol=1e-7)
    np.testing.assert_allclose(rotated.carrier_density_um3,fine.carrier_density_um3,rtol=1e-9)
