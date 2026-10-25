"""Open-boundary, lossless multilayer TE modes; z in um, k0/beta in um^-1.

Theta'' + (k0**2*epsilon-beta**2)*Theta = 0; exp(+i*omega*t),
exp(-i*beta*x). Complex epsilon is deliberately rejected: under this convention
passivity requires Im(epsilon)<0 and forward attenuation Im(beta)<0. Complex
field amplitudes are supported, not complex-material eigenproblems.

Exact element Dirichlet-to-Neumann maps form a symmetric tridiagonal matrix.
Oscillatory layers are split so their maximum phase is <= pi/3 (no element
poles). Its eigenvalues increase strictly with beta**2 by the variational
identity dK/d(beta**2) = integral(N_i*N_j); cladding Robin terms also increase.
Thus eigenvalue-index brackets, not determinant sign scans, enumerate modes.
Evanescent maps use only exp(-q*d), never growing exponentials. No finite box.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


class NoGuidedModeError(ValueError):
    """No strictly bound TE mode exists at the requested index/frequency."""


def _positive_scalar(value, name):
    a = np.asarray(value)
    if a.ndim or a.dtype.kind not in 'iuf' or not np.isfinite(a) or a <= 0:
        raise ValueError(f'{name} must be a finite positive real scalar')
    return float(a)


def _basis(s, length, delta):
    """Endpoint shape functions and their x derivatives; x=k0*z."""
    s = np.asarray(s, dtype=float)
    t = np.sqrt(abs(delta))*length
    if t < 1e-5:
        # sinc/sinhc expansion gives the exact delta=0 linear limit.
        u = s/length
        def h(v):
            return v*(1-delta*(length*v)**2/6+delta**2*(length*v)**4/120)
        den = 1-delta*length**2/6+delta**2*length**4/120
        left, right = h(1-u)/den, h(u)/den
        def dh(v):
            return (1-delta*(length*v)**2/2+delta**2*(length*v)**4/24)/(length*den)
        return left, right, -dh(1-u), dh(u)
    q = np.sqrt(abs(delta))
    if delta > 0:
        den = np.sin(t)
        return (np.sin(q*(length-s))/den, np.sin(q*s)/den,
                -q*np.cos(q*(length-s))/den, q*np.cos(q*s)/den)
    den = -np.expm1(-2*t)
    def h(v):
        return np.exp(-q*(length-v))*(-np.expm1(-2*q*v))/den
    def dh(v):
        return q*np.exp(-q*(length-v))*(1+np.exp(-2*q*v))/den
    return h(length-s), h(s), -dh(length-s), dh(s)


def _element(length, delta):
    t = np.sqrt(abs(delta))*length
    if t < 1e-5:
        diag = (1-delta*length**2/3-delta**2*length**4/45)/length
        off = -(1+delta*length**2/6+7*delta**2*length**4/360)/length
    elif delta > 0:
        p = np.sqrt(delta)
        diag, off = p/np.tan(t), -p/np.sin(t)
    else:
        q = np.sqrt(-delta)
        e = np.exp(-t)
        den = -np.expm1(-2*t)
        diag, off = q*(1+e*e)/den, -2*q*e/den
    return diag, off


def _mass(length, delta):
    t = np.sqrt(abs(delta))*length
    if t < .1:
        # Smooth small-phase quadrature avoids cancellation in analytic forms.
        x, w = np.polynomial.legendre.leggauss(12)
        a, b, _, _ = _basis((x+1)*length/2, length, delta)
        return length/2*np.dot(w, a*a), length/2*np.dot(w, a*b)
    q = np.sqrt(abs(delta))
    if delta > 0:
        return ((length/2-np.sin(2*t)/(4*q))/np.sin(t)**2,
                (np.sin(t)/(2*q)-length*np.cos(t)/2)/np.sin(t)**2)
    e = np.exp(-t)
    csch = 2*e/(-np.expm1(-2*t))
    coth = (1+e*e)/(-np.expm1(-2*t))
    return (coth-t*csch*csch)/(2*q), (t*coth-1)*csch/(2*q)


class TEWaveguide:
    """eps=[lower halfspace, finite layers..., upper halfspace].

    thickness_um contains ONLY positive finite internal thicknesses. z=0 is
    the first interface; an interface's epsilon belongs to its right layer.
    Exactly one of k0 or wavelength_um is required: k0=2*pi/wavelength_um.
    period_um is optional, fixed metadata; never inferred from beta or changed.
    Equal adjacent epsilon is legal; zero thickness must be removed by caller.
    Configuration is read-only; construct a new guide to change it. Public
    sequences are immutable snapshots, consistent with the derived mesh.
    """
    def __init__(self, eps, thickness_um, *, k0=None, wavelength_um=None, period_um=None):
        e = np.asarray(eps)
        d = np.asarray(thickness_um)
        if e.ndim != 1 or e.size < 3 or e.dtype.kind not in 'iufc':
            raise ValueError('eps must contain two halfspaces and at least one finite layer')
        if not np.all(np.isfinite(e)) or np.any(e.imag != 0) or np.any(e.real <= 0):
            raise ValueError('Only finite positive lossless relative epsilon is supported')
        if d.ndim != 1 or d.size != e.size-2 or d.dtype.kind not in 'iuf':
            raise ValueError('thickness_um must describe exactly the finite layers')
        if not np.all(np.isfinite(d)) or np.any(d <= 0):
            raise ValueError('Finite layer thicknesses must be positive and finite')
        if (k0 is None) == (wavelength_um is None):
            raise ValueError('Specify exactly one of k0 and wavelength_um')
        self._k0 = (_positive_scalar(k0, 'k0') if k0 is not None else
                   2*np.pi/_positive_scalar(wavelength_um, 'wavelength_um'))
        if not np.isfinite(self.k0):
            raise ValueError('Derived k0 is not representable')
        self._period_um = None if period_um is None else _positive_scalar(period_um, 'period_um')
        self._eps = tuple(float(v.real) for v in e)
        self._thickness_um = tuple(float(v) for v in d)
        self._interfaces_um = tuple(np.r_[0., np.cumsum(d)])
        self._floor = max(self.eps[0], self.eps[-1])
        lengths, materials = [], []
        for epsilon, width in zip(self.eps[1:-1], self.thickness_um):
            optical = self.k0*width
            n = max(1, int(np.ceil(optical*np.sqrt(max(0., epsilon-self._floor))/(np.pi/3))))
            if n > 20000:
                raise ValueError('Optical thickness exceeds supported element budget')
            lengths.extend([optical/n]*n)
            materials.extend([epsilon]*n)
        self._lengths = np.array(lengths)
        self._materials = np.array(materials)
        self._nodes = np.r_[0., np.cumsum(self._lengths)]/self.k0

    @property
    def k0(self):
        return self._k0

    @property
    def period_um(self):
        return self._period_um

    @property
    def eps(self):
        return self._eps

    @property
    def thickness_um(self):
        return self._thickness_um

    @property
    def interfaces_um(self):
        return self._interfaces_um

    def eps_profile(self, z_um):
        z = _coordinates(z_um)
        return np.asarray(self.eps)[np.searchsorted(self.interfaces_um, z, side='right')]

    def _matrix(self, value):
        pairs = [_element(d, e-value) for d, e in zip(self._lengths, self._materials)]
        diag = np.zeros(len(pairs)+1)
        off = np.array([p[1] for p in pairs])
        for i, (a, _) in enumerate(pairs):
            diag[i] += a
            diag[i+1] += a
        diag[0] += np.sqrt(value-self.eps[0])
        diag[-1] += np.sqrt(value-self.eps[-1])
        return diag, off

    def solve_modes(self):
        """All strictly bound modes sorted by decreasing beta (fundamental first).

        Modes within 1e-12 in effective-index squared of the cladding threshold
        or numerically unresolved degeneracies are not certified. A candidate
        failing the original endpoint derivative residual raises RuntimeError.
        """
        lo, hi = self._floor, max(self.eps)
        if hi <= lo:
            return []
        vals = eigh_tridiagonal(*self._matrix(lo), eigvals_only=True)
        count = int(np.count_nonzero(vals < -1e-12))
        modes = []
        for i in range(count):
            def eigenvalue(v):
                return eigh_tridiagonal(*self._matrix(v), select='i',
                                        select_range=(i, i), eigvals_only=True)[0]
            root = brentq(eigenvalue, lo, hi, xtol=5e-15, rtol=1e-14)
            if root-lo <= 1e-12:
                continue
            diag, off = self._matrix(root)
            _, vectors = eigh_tridiagonal(diag, off, select='i', select_range=(i, i))
            mode = TEMode(self, root, vectors[:, 0])
            if mode.boundary_residual > 1e-8:
                raise RuntimeError('Candidate fails original interface/boundary residual')
            modes.append(mode)
        modes.sort(key=lambda m: m.beta, reverse=True)
        if any(abs(a.beta-b.beta) < 1e-11*self.k0 for a, b in zip(modes, modes[1:])):
            raise RuntimeError('Numerically unresolved near-degenerate modes')
        return modes

    def solve_mode(self, index=0):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError('Mode index must be an integer')
        if index < 0:
            raise ValueError('Mode index must be nonnegative')
        modes = self.solve_modes()
        if not modes:
            raise NoGuidedModeError('No strictly bound TE guided mode')
        if index >= len(modes):
            raise IndexError('Guided mode index out of range')
        return modes[index]


def _coordinates(z):
    a = np.asarray(z)
    if a.dtype.kind not in 'iuf' or not np.all(np.isfinite(a)):
        raise ValueError('z_um must be finite real coordinates')
    return a.astype(float)


class TEMode:
    """Complex-valued normalized field: integral over all z |theta|^2 dz = 1.

    theta/Theta0 and derivative broadcast z and an optional complex amplitude.
    A non-unit amplitude intentionally changes the norm to |amplitude|^2.
    guide, beta and boundary_residual are read-only solved-state metadata.
    """
    def __init__(self, guide, value, nodes):
        self._guide = guide
        self._beta = guide.k0*np.sqrt(value)
        self._value = value
        nodes = nodes.copy()
        if nodes[np.argmax(abs(nodes))] < 0:
            nodes *= -1
        gamma = np.sqrt(value-np.array([guide.eps[0], guide.eps[-1]]))
        norm = nodes[0]**2/(2*gamma[0])+nodes[-1]**2/(2*gamma[1])
        for i, (d, e) in enumerate(zip(guide._lengths, guide._materials)):
            a, b = _mass(d, e-value)
            norm += a*(nodes[i]**2+nodes[i+1]**2)+2*b*nodes[i]*nodes[i+1]
        self._field = nodes*np.sqrt(guide.k0/norm)
        # Check original endpoint derivatives, not optimizer status/determinant.
        starts, ends = [], []
        for i, (d, e) in enumerate(zip(guide._lengths, guide._materials)):
            _, _, a, b = _basis(np.array([0., d]), d, e-value)
            v = a*self._field[i]+b*self._field[i+1]
            starts.append(v[0])
            ends.append(v[1])
        residual = np.r_[starts[0]-gamma[0]*self._field[0],
                         np.array(starts[1:])-np.array(ends[:-1]),
                         ends[-1]+gamma[1]*self._field[-1]]
        self._boundary_residual = float(np.max(abs(residual))/
                                       (max(abs(self._field))*max(1., np.sqrt(value))))

    @property
    def guide(self):
        return self._guide

    @property
    def beta(self):
        return self._beta

    @property
    def boundary_residual(self):
        return self._boundary_residual

    def _evaluate(self, z_um, derivative, amplitude):
        z, amp = np.broadcast_arrays(_coordinates(z_um), np.asarray(amplitude, dtype=complex))
        if not np.all(np.isfinite(amp)):
            raise ValueError('amplitude must be finite')
        g = self.guide
        flat = z.ravel()
        out = np.empty(flat.size, dtype=complex)
        indices = np.searchsorted(g._nodes, flat, side='right')-1
        for i in np.unique(indices):
            mask = indices == i
            if i < 0 or i >= len(g._lengths):
                lower = i < 0
                boundary = 0 if lower else -1
                gamma = g.k0*np.sqrt(self._value-g.eps[boundary])
                sign = 1 if lower else -1
                out[mask] = self._field[boundary]*np.exp(sign*gamma*(flat[mask]-g._nodes[boundary]))
                if derivative:
                    out[mask] *= sign*gamma
            else:
                a, b, da, db = _basis((flat[mask]-g._nodes[i])*g.k0,
                                      g._lengths[i], g._materials[i]-self._value)
                out[mask] = ((da*self._field[i]+db*self._field[i+1])*g.k0 if derivative
                             else a*self._field[i]+b*self._field[i+1])
        result = out.reshape(z.shape)*amp
        return result.item() if result.ndim == 0 else result

    def theta(self, z_um, amplitude=1):
        return self._evaluate(z_um, False, amplitude)

    Theta0 = theta

    def derivative(self, z_um, amplitude=1):
        return self._evaluate(z_um, True, amplitude)

    def eps_profile(self, z_um):
        return self.guide.eps_profile(z_um)
