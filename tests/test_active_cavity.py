"""Independent physical gates for the active adapter (um, ns, photons)."""
import numpy as np
import pytest
from scipy.constants import c, hbar
from cwt3d.finite.assembly import assemble_pencil
from cwt3d.finite.grid import StaggeredGrid
from cwt3d.active.cavity import ActiveCavity


class Material:
    def gain(self, N, U):
        return .03 * N / (1 + .01 * U)

    def index_change(self, N):
        return .002 * N


def cavity(C=None, **kwargs):
    p = assemble_pencil(StaggeredGrid(3, 2, 6., 3.),
                        np.zeros((4, 4)) if C is None else C)
    options = dict(group_index=3.5, k0_um_inverse=6., active_thickness_um=.1,
                   confinement=.2)
    options.update(kwargs)
    return ActiveCavity(p, **options)


def vector(a):
    r = np.random.default_rng(22)
    return r.normal(size=a.grid.n_dof) + 1j*r.normal(size=a.grid.n_dof)


def test_photon_number_density_and_seed_are_physical():
    a = cavity()
    v = vector(a)
    q = (a.B @ v).reshape(4, 2, 3)
    S = np.sum(abs(q)**2, axis=0)
    area = a.grid.hx_um*a.grid.hy_um
    np.testing.assert_allclose(a.center_fields(v), q)
    np.testing.assert_allclose(a.photon_density(v), a.confinement/.1*S)
    assert a.photon_number(v) == pytest.approx(area*np.sum(S))
    seed = a.normalize_seed(v, 127.)
    assert a.photon_number(seed) == pytest.approx(127.)
    assert np.sum(a.photon_density(seed))*area*.1/a.confinement == pytest.approx(127.)
    np.testing.assert_array_equal(a.normalize_seed(v, 0), 0)
    assert a.vg_um_ns == pytest.approx(c*1e-3/3.5)
    assert a.photon_energy_j == pytest.approx(hbar*c*6e6)


def test_passive_recovery_and_patterned_material_operator():
    a = cavity(internal_loss_um_inverse=np.arange(6).reshape(2,3)*.001)
    N = np.arange(6).reshape(2, 3)/2
    U = np.ones((2,3))*7
    mat = Material()
    g, dn = mat.gain(N,U), mat.index_change(N)
    shift = -6*.2*dn - .5j*(.2*g-a.internal_loss_um_inverse)
    A,B = a.frozen_pencil(N,mat,photon_density_um3=U)
    np.testing.assert_allclose((A-a.A).toarray(), np.diag(np.tile(shift.ravel(),4))@B.toarray())
    np.testing.assert_allclose(a.material_rate(N,U,mat),
        (a.vg_um_ns*((.2*g-a.internal_loss_um_inverse)/2-1j*6*.2*dn)).ravel())
    z = cavity()
    A,B = z.frozen_pencil(0,mat)
    np.testing.assert_array_equal(A.toarray(),z.A.toarray())
    np.testing.assert_array_equal(B.toarray(),z.B.toarray())
    # Newton trial densities may be negative, public frozen state may not.
    assert np.all(np.isfinite(z.material_rate(-1,0,mat)))
    with pytest.raises(ValueError): z.frozen_pencil(-1,mat)


@pytest.mark.parametrize('generic_gain', [False, True])
def test_exact_arbitrary_field_power_identity(generic_gain):
    r = np.random.default_rng(8)
    C = r.normal(size=(4,4)) + 1j*r.normal(size=(4,4))
    if not generic_gain: C = (C+C.conj().T)/2
    a = cavity(C, internal_loss_um_inverse=.013)
    v = vector(a)
    N = np.arange(6).reshape(2,3)+1
    q = a.B@v
    U = a.photon_density(v)
    m = np.tile(a.material_rate(N,U,Material()),4)
    # From B dv/dt = i vg A0 v + m Bv; no B inverse is needed.
    dq = 1j*a.vg_um_ns*(a.A@v) + m*q
    derivative = 2*a.grid.hx_um*a.grid.hy_um*np.vdot(q,dq).real
    b = a.power_budget(v,N,Material())
    assert b['dphotons_dt_per_ns'] == pytest.approx(derivative,abs=1e-7)
    assert b['total_photons'] == pytest.approx(a.photon_number(v))
    assert b['coupling_exchange_w'] == pytest.approx(b['coupling_loss_per_ns']*a.photon_energy_j*1e9)
    assert 'radiative_output_w' not in b
    f = a.grid.unpack(v)
    edge = a.vg_um_ns*(a.grid.hy_um*(np.sum(abs(f.Rx[:,-1])**2)+np.sum(abs(f.Sx[:,0])**2))
            +a.grid.hx_um*(np.sum(abs(f.Ry[-1,:])**2)+np.sum(abs(f.Sy[0,:])**2)))
    assert b['edge_loss_per_ns'] == pytest.approx(edge)


def test_snapshots_do_not_expose_shared_headers():
    alpha = np.full((2,3),.01)
    a = cavity(internal_loss_um_inverse=alpha)
    alpha[:] = 42
    np.testing.assert_array_equal(a.internal_loss_um_inverse,.01)
    x = a.internal_loss_um_inverse
    assert isinstance(x.base,bytes)
    x.shape=(6,)
    assert a.internal_loss_um_inverse.shape == (2,3)
    for name in ('A','B','D'):
        old = getattr(a,name).toarray()
        getattr(a,name).data[:] = 999
        np.testing.assert_array_equal(getattr(a,name).toarray(),old)
    for x in (a.center_fields(vector(a)), a.photon_density(vector(a)),
              a.normalize_seed(vector(a),1),a.material_rate(1,0,Material())):
        assert isinstance(x.base,bytes)
        with pytest.raises(ValueError): x.flat[0]=0
    with pytest.raises((AttributeError,TypeError)): a.confinement=.9


@pytest.mark.parametrize('key,value', [('group_index',0),('k0_um_inverse',-1),
    ('active_thickness_um',np.inf),('confinement',1.1),('confinement',0),
    ('group_index',True),('internal_loss_um_inverse',-1),
    ('internal_loss_um_inverse',np.ones((1,3)))])
def test_invalid_constructor(key,value):
    with pytest.raises(ValueError): cavity(**{key:value})


@pytest.mark.parametrize('N,U', [(-1,0),(1,-1),(np.ones((3,2)),0),
    (np.ones((1,2,3)),0),(np.nan,0),(1,np.inf),(1,np.ones((1,3)))])
def test_invalid_frozen_state(N,U):
    with pytest.raises(ValueError): cavity().frozen_pencil(N,Material(),photon_density_um3=U)


def test_material_result_cannot_expand_grid():
    class Bad(Material):
        def gain(self,N,U): return np.zeros((1,2,3))
    with pytest.raises(ValueError): cavity().material_rate(1,0,Bad())
    class Row(Material):
        def gain(self,N,U): return np.ones((1,3))
    assert cavity().material_rate(1,0,Row()).shape == (6,)


@pytest.mark.parametrize('photons',[1,-1,np.nan,True])
def test_bad_seed(photons):
    with pytest.raises(ValueError): cavity().normalize_seed(np.zeros(24),photons)


@pytest.mark.parametrize('all_evanescent', [True,False])
def test_optional_radiation_only_uses_crad(all_evanescent):
    from cwt3d.coupling import CouplingResult, _pack
    rad = .03j*np.eye(4)
    other = .02j*np.eye(4)
    C = rad+other
    coupling = CouplingResult(_pack(C=C,C1D=other,Crad=rad,C2D=np.zeros((4,4))),
        period_um=1,beta0=6,k0=6,relative_detuning=0,cutoff=1,
        quadrature_order=2,pc_layers=(),radial_orders=(),all_aux_evanescent=all_evanescent)
    a=cavity(C,coupling=coupling)
    v=a.normalize_seed(vector(a),10)
    b=a.power_budget(v,0,Material())
    assert b['radiative_output_w'] == pytest.approx(a.vg_um_ns*2*.03*10*a.photon_energy_j*1e9)
    assert b['radiative_output_w'] != pytest.approx(b['coupling_exchange_w'],abs=1e-20)
    assert ('all radiative' in a.radiation_scope) == all_evanescent
    with pytest.raises(ValueError): cavity(np.zeros((4,4)),coupling=coupling)
    with pytest.raises(TypeError): cavity(C,coupling=object())


def physical_coupling(k0=6.):
    from cwt3d.coupling import CouplingResult, _pack
    rad = .03j*np.eye(4)
    return CouplingResult(
        _pack(C=rad, C1D=np.zeros((4,4)), Crad=rad, C2D=np.zeros((4,4))),
        period_um=1, beta0=6, k0=k0, relative_detuning=0, cutoff=1,
        quadrature_order=2, pc_layers=(), radial_orders=(), all_aux_evanescent=True)


@pytest.mark.parametrize('declared_k0', [600., 6.000001, np.nextafter(6., np.inf)])
def test_physical_coupling_requires_exact_reference_wavenumber(declared_k0):
    coupling = physical_coupling()
    with pytest.raises(ValueError, match='coupling.k0'):
        cavity(coupling.C, coupling=coupling, k0_um_inverse=declared_k0)


@pytest.mark.parametrize('source_k0', [0., -6., np.nan, np.inf, -np.inf,
                                       True, 6+0j, '6', None])
def test_physical_coupling_requires_finite_positive_real_wavenumber(source_k0):
    coupling = physical_coupling(source_k0)
    with pytest.raises(ValueError, match='coupling.k0'):
        cavity(coupling.C, coupling=coupling)


@pytest.mark.parametrize('k0', [6., np.float64(6.)])
def test_physical_reference_binds_energy_index_and_radiation(k0):
    coupling = physical_coupling(k0)
    a = cavity(coupling.C, coupling=coupling, k0_um_inverse=k0)
    assert a.photon_energy_j == pytest.approx(hbar*c*float(k0)*1e6, rel=1e-14, abs=0)
    mat = Material()
    rate = a.material_rate(2., 0., mat)
    np.testing.assert_allclose(rate.imag, -a.vg_um_ns*float(k0)*a.confinement*.004)
    v = a.normalize_seed(vector(a), 10.)
    budget = a.power_budget(v, 2., mat)
    expected = 2*a.vg_um_ns*.03*10*hbar*c*float(k0)*1e6*1e9
    assert budget['radiative_output_w'] == pytest.approx(expected, rel=1e-14, abs=0)


def test_generic_coupling_has_no_reference_wavenumber_binding():
    a = cavity(.03j*np.eye(4), k0_um_inverse=600.)
    assert a.photon_energy_j == pytest.approx(hbar*c*600e6, rel=1e-14, abs=0)
    assert 'radiative_output_w' not in a.power_budget(vector(a), 0., Material())


def test_uniform_volume_stimulated_gain_balance():
    class Constant(Material):
        def gain(self,N,U): return .04
        def index_change(self,N): return 0
    a=cavity(internal_loss_um_inverse=.003)
    v=a.normalize_seed(vector(a),19.)
    b=a.power_budget(v,1,Constant())
    assert b['stimulated_per_ns'] == pytest.approx(a.vg_um_ns*.2*.04*19)
    assert b['internal_loss_per_ns'] == pytest.approx(a.vg_um_ns*.003*19)
    assert b['dphotons_dt_per_ns'] == pytest.approx(
        a.vg_um_ns*(.2*.04-.003)*19-b['edge_loss_per_ns'])


@pytest.mark.parametrize('scale',[1e-300,1e300])
def test_seed_arbitrary_scale_removed_before_squaring(scale):
    a=cavity()
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        v=a.normalize_seed(vector(a)*scale,3)
    assert a.photon_number(v) == pytest.approx(3)


def test_real_material_scalar_and_patterned_maps_integrate():
    from cwt3d.active.materials import CarrierMaterial
    a=cavity()
    for differential_gain in (.04,np.arange(6).reshape(2,3)*.001+.04):
        mat=CarrierMaterial(differential_gain,2.,gain_compression_um3=.01,
                            index_per_density_um3=-.002)
        N=np.arange(6).reshape(2,3)+3.
        U=np.full((2,3),4.)
        expected=a.vg_um_ns*((a.confinement*mat.gain(N,U))/2
                 -1j*a.k0_um_inverse*a.confinement*mat.index_change(N))
        np.testing.assert_allclose(a.material_rate(N,U,mat),expected.ravel())
        A,B=a.frozen_pencil(N,mat,photon_density_um3=U)
        assert A.shape==B.shape==(24,24)


def test_wrong_vector_and_pencil_types():
    with pytest.raises(TypeError): ActiveCavity(object(),group_index=3,k0_um_inverse=6,active_thickness_um=.1,confinement=.2)
    for v in (np.zeros((24,1)),np.full(24,np.nan)):
        with pytest.raises(ValueError): cavity().photon_number(v)
