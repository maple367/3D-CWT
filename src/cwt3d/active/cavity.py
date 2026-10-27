"""Sparse active adapter in um, ns and photons; generic coupling stays signed."""
from dataclasses import dataclass
from numbers import Real
import numpy as np
from scipy.constants import c, hbar
from scipy.sparse import diags
from ..finite.assembly import FinitePencil
from ..finite.grid import _complex_array, _snapshot


def _scalar(value, name, *, positive=False, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real scalar')
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} must be finite and physically bounded')
    if upper is not None and value > upper:
        raise ValueError(f'{name} must not exceed {upper}')
    return value


def _real(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must contain finite real values')
    return np.asarray(a, dtype=float)


@dataclass(frozen=True, init=False, eq=False)
class ActiveCavity:
    """Immutable optical adapter with independent public sparse/array snapshots.

    Negative carrier Newton trials are allowed in material_rate only. Frozen
    spectra require physical densities. Only seeds are normalized, never
    propagated states. No dense matrix or inverse of B is constructed.
    """
    grid: object
    group_index: float
    k0_um_inverse: float
    active_thickness_um: float
    confinement: float
    vg_um_ns: float
    photon_energy_j: float
    radiation_scope: str
    _A: object
    _B: object
    _D: object
    _c_bytes: bytes
    _loss_bytes: bytes
    _rad_bytes: object

    def __init__(self, pencil, *, group_index, k0_um_inverse,
                 active_thickness_um, confinement, internal_loss_um_inverse=0.,
                 coupling=None):
        if not isinstance(pencil, FinitePencil):
            raise TypeError('pencil must be a FinitePencil')
        ng = _scalar(group_index, 'group_index', positive=True)
        k0 = _scalar(k0_um_inverse, 'k0_um_inverse', positive=True)
        d = _scalar(active_thickness_um, 'active_thickness_um', positive=True)
        gamma = _scalar(confinement, 'confinement', positive=True, upper=1)
        shape = (pencil.grid.Ny, pencil.grid.Nx)
        loss = _real(internal_loss_um_inverse, 'internal_loss_um_inverse')
        if loss.shape not in ((), shape) or np.any(loss < 0):
            raise ValueError('internal loss must be nonnegative scalar or exact grid array')
        loss = np.broadcast_to(loss, shape)
        C = pencil.C
        rad = None
        scope = 'unspecified: generic coupling matrix'
        if coupling is not None:
            from ..coupling import CouplingResult
            if not isinstance(coupling, CouplingResult):
                raise TypeError('coupling must be a CouplingResult')
            source_k0 = _scalar(coupling.k0, 'coupling.k0', positive=True)
            # One physical reference frequency: no detuning/roundoff allowance.
            if source_k0 != k0:
                raise ValueError('coupling.k0 must exactly match k0_um_inverse')
            if coupling.units != 'um^-1' or not np.array_equal(coupling.C, C):
                raise ValueError('coupling must exactly match pencil.C and units')
            parts = [_complex_array(getattr(coupling, n), n)
                     for n in ('C1D', 'Crad', 'C2D')]
            if any(p.shape != (4,4) for p in parts):
                raise ValueError('coupling components must have shape (4,4)')
            scale = max(float(np.max(abs(p))) for p in parts)
            if ((scale and np.max(abs(sum(p/scale for p in parts)-C/scale)) > 64*np.finfo(float).eps)
                    or (not scale and np.any(C))):
                raise ValueError('coupling components must sum to pencil.C')
            rad = parts[1].tobytes()
            scope = ('all radiative coupling (auxiliary orders evanescent)'
                     if coupling.all_aux_evanescent else
                     'zero-order only; auxiliary orders may radiate')
        vg, energy = c*1e-3/ng, hbar*c*k0*1e6
        area = pencil.grid.hx_um*pencil.grid.hy_um
        if not all(np.isfinite(x) and x > 0 for x in (vg, energy, area, gamma/d)):
            raise ValueError('derived optical scales outside finite numeric range')
        values = dict(grid=pencil.grid, group_index=ng, k0_um_inverse=k0,
                      active_thickness_um=d, confinement=gamma, vg_um_ns=vg,
                      photon_energy_j=energy, radiation_scope=scope,
                      _A=pencil.A, _B=pencil.B, _D=pencil.D,
                      _c_bytes=C.tobytes(), _loss_bytes=loss.tobytes(), _rad_bytes=rad)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def A(self):
        return self._A.copy()

    @property
    def B(self):
        return self._B.copy()

    @property
    def D(self):
        return self._D.copy()

    @property
    def C(self):
        return np.ndarray((4,4), dtype=complex, buffer=self._c_bytes)

    @property
    def internal_loss_um_inverse(self):
        return np.ndarray(self._shape, dtype=float, buffer=self._loss_bytes)

    @property
    def _shape(self):
        return (self.grid.Ny, self.grid.Nx)

    @property
    def _area(self):
        return self.grid.hx_um*self.grid.hy_um

    def _vector(self, v):
        v = _complex_array(v, 'vector')
        if v.shape != (self.grid.n_dof,):
            raise ValueError(f'vector must have shape ({self.grid.n_dof},)')
        return v

    def _state(self, value, name, *, nonnegative=True):
        a = _real(value, name)
        if a.shape not in ((), self._shape) or (nonnegative and np.any(a < 0)):
            raise ValueError(f'{name} must be scalar or exact grid array with physical bounds')
        return np.broadcast_to(a, self._shape)

    def _material(self, N, U, material, *, physical):
        N = self._state(N, 'carrier density', nonnegative=physical)
        U = self._state(U, 'photon density')
        results = []
        for value, name in ((material.gain(N,U), 'material gain'),
                            (material.index_change(N), 'material index change')):
            a = _real(value, name)
            try:
                results.append(np.broadcast_to(a, self._shape))
            except ValueError as exc:
                raise ValueError(f'{name} must broadcast to the grid without expanding it') from exc
        return results

    def center_fields(self, v):
        """Read-only component-major (4,Ny,Nx) center amplitudes Bv."""
        q = self._B @ self._vector(v)
        if not np.all(np.isfinite(q)):
            raise ValueError('center amplitudes outside finite numeric range')
        return _snapshot(q.reshape((4,)+self._shape))

    def photon_density(self, v):
        """Active-volume density U=Gamma/d * sum_components |Bv|²."""
        s = self.confinement/self.active_thickness_um*np.sum(abs(self.center_fields(v))**2,axis=0)
        if not np.all(np.isfinite(s)):
            raise ValueError('photon density outside finite numeric range')
        return _snapshot(s)

    def photon_number(self, v):
        p = float(self._area*np.sum(abs(self.center_fields(v))**2))
        if not np.isfinite(p):
            raise ValueError('photon number outside finite numeric range')
        return p

    def normalize_seed(self, v, photons):
        """Normalize only the initial seed to a physical total photon count."""
        photons = _scalar(photons, 'photons')
        v = self._vector(v)
        if photons == 0:
            return _snapshot(np.zeros_like(v))
        scale = max(float(np.max(abs(v.real))), float(np.max(abs(v.imag))))
        if not scale:
            raise ValueError('positive photon count requires nonzero seed')
        w = np.empty_like(v)
        w.real, w.imag = v.real/scale, v.imag/scale
        p = self.photon_number(w)
        if p <= 0:
            raise ValueError('seed has zero center photon number')
        w *= np.sqrt(photons)/np.sqrt(p)
        if not np.all(np.isfinite(w)):
            raise ValueError('normalized seed outside finite numeric range')
        return _snapshot(w)

    def material_rate(self, N, U, material):
        """Flat ncell local complex temporal rate, ns^-1; Newton trials allowed."""
        g, dn = self._material(N, U, material, physical=False)
        m = self.vg_um_ns*((self.confinement*g-self.internal_loss_um_inverse)/2
                          -1j*self.k0_um_inverse*self.confinement*dn)
        if not np.all(np.isfinite(m)):
            raise ValueError('material rate outside finite numeric range')
        return _snapshot(m.ravel())

    def frozen_pencil(self, N, material, *, photon_density_um3=0.):
        """Freeze material without changing sparse transport or boundaries."""
        g, dn = self._material(N, photon_density_um3, material, physical=True)
        shift = (-self.k0_um_inverse*self.confinement*dn
                 -.5j*(self.confinement*g-self.internal_loss_um_inverse))
        if not np.all(np.isfinite(shift)):
            raise ValueError('frozen shift outside finite numeric range')
        a = self._A + diags(np.tile(shift.ravel(),4), format='csr') @ self._B
        if not np.all(np.isfinite(a.data)):
            raise ValueError('frozen pencil outside finite numeric range')
        return a, self._B.copy()

    def power_budget(self, v, N, material):
        """Photon rates and signed coupling watts; edge escape is not vertical radiation."""
        v = self._vector(v)
        q = self.center_fields(v)
        s = np.sum(abs(q)**2, axis=0)
        U = self.confinement/self.active_thickness_um*s
        gain, _ = self._material(N,U,material,physical=True)
        f = self.grid.unpack(v)
        edge = self.vg_um_ns*(self.grid.hy_um*(np.sum(abs(f.Rx[:,-1])**2)+np.sum(abs(f.Sx[:,0])**2))
                 +self.grid.hx_um*(np.sum(abs(f.Ry[-1,:])**2)+np.sum(abs(f.Sy[0,:])**2)))
        qc = q.reshape(4,-1)
        def exchange(C):
            return float(2*self.vg_um_ns*self._area*np.vdot(qc,C@qc).imag)
        coupling = exchange(self.C)
        internal = float(self.vg_um_ns*self._area*np.sum(self.internal_loss_um_inverse*s))
        stim = float(self.vg_um_ns*self.confinement*self._area*np.sum(gain*s))
        result = dict(total_photons=float(self._area*np.sum(s)),
                      edge_loss_per_ns=float(edge), coupling_loss_per_ns=coupling,
                      internal_loss_per_ns=internal, stimulated_per_ns=stim,
                      dphotons_dt_per_ns=float(stim-internal-edge-coupling),
                      coupling_exchange_w=coupling*self.photon_energy_j*1e9)
        if self._rad_bytes is not None:
            Crad = np.ndarray((4,4),dtype=complex,buffer=self._rad_bytes)
            result['radiative_output_w'] = exchange(Crad)*self.photon_energy_j*1e9
        if not all(np.isfinite(x) for x in result.values()):
            raise ValueError('power budget outside finite numeric range')
        return result
