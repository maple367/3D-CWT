"""Outgoing scalar Green kernel of an infinite homogeneous passive medium.

Lengths are um, k0 and k_parallel are um^-1, epsilon is relative permittivity.
With exp(+i*omega*t), passive media have Im(epsilon) <= 0. This solves

    [d_z^2 + k0^2*epsilon - k_parallel^2] G(z,z') = -delta(z-z').

Set gamma = sqrt(k_parallel^2-k0^2*epsilon), on the branch Re(gamma)>=0
and Im(gamma)>=0 (including the lossless propagating boundary). Then
G=exp(-gamma*abs(z-z'))/(2*gamma). Equivalently kz=-i*gamma has Re(kz)>=0,
Im(kz)<=0 and G=-i*exp(-i*kz*abs(z-z'))/(2*kz). The derivative jump at the
source is -1. No growing exponential or legacy exp(i*gamma*d) is used.

HomogeneousGreen and LayeredGreen implement scalar TE kernels, not TM
matching, harmonic enumeration or a coupling matrix.
"""
from __future__ import annotations

import cmath
import math

import numpy as np


class CutoffError(ValueError):
    """No finite unique outgoing kernel at numerically exact gamma=0."""


def _contains_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_bool(item) for item in value)
    return isinstance(value, np.ndarray) and value.dtype.kind == 'b'


def _numeric(value, name, *, complex_allowed=False):
    if _contains_bool(value):
        raise ValueError(f'{name} must be numeric, not boolean')
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite numeric value') from exc
    kinds = 'iufc' if complex_allowed else 'iuf'
    if array.dtype.kind not in kinds or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite and ' + ('numeric' if complex_allowed else 'real'))
    return array


def _scalar(value, name, *, complex_allowed=False):
    array = _numeric(value, name, complex_allowed=complex_allowed)
    if array.ndim != 0:
        raise ValueError(f'{name} must be scalar')
    result = complex(array) if complex_allowed else float(array)
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError(f'{name} is outside finite float64 range')
    return result


class HomogeneousGreen:
    """Read-only configuration for an infinite homogeneous Green kernel.

    Parameters: finite real k0>0, finite passive complex epsilon, and finite
    real k_parallel>=0 (default zero). All are scalar; to change configuration
    construct a new kernel. Zero-dimensional NumPy inputs are copied to Python
    scalars, so there is no mutable array or metadata alias.

    Coordinates are finite real scalars/arrays in um and follow NumPy
    broadcasting. For two independent grids use z[:,None], z_prime[None,:].
    Outputs are complex with the broadcast shape; evaluation has no cache.

    Exact float64 cutoff raises CutoffError, including when subtraction has
    rounded an unresolved near-cutoff case to zero. Nonzero gamma is never
    tolerance-clamped. Extreme parameters/intermediates outside float64 range
    are rejected, not claimed to have a resolved physical cutoff.
    """

    __slots__ = ('_k0', '_epsilon', '_k_parallel', '_gamma')

    def __init__(self, k0, epsilon, k_parallel=0):
        k0 = _scalar(k0, 'k0')
        epsilon = _scalar(epsilon, 'epsilon', complex_allowed=True)
        k_parallel = _scalar(k_parallel, 'k_parallel')
        if k0 <= 0 or k_parallel < 0:
            raise ValueError('k0 must be positive and k_parallel nonnegative')
        if epsilon.imag > 0:
            raise ValueError('active epsilon is unsupported: exp(+i*omega*t) requires Im(epsilon)<=0')
        k0_squared = k0 * k0
        parallel_squared = k_parallel * k_parallel
        if not math.isfinite(k0_squared) or not math.isfinite(parallel_squared) or k0_squared == 0:
            raise ValueError('squared wavenumber outside resolved float64 range')
        radicand = parallel_squared - k0_squared * epsilon
        if not (math.isfinite(radicand.real) and math.isfinite(radicand.imag)):
            raise ValueError('derived gamma squared outside finite float64 range')
        if radicand == 0:
            raise CutoffError('numerically exact cutoff: gamma squared is zero; exact or unresolved near-cutoff')
        # The passive radicand lies in the upper half plane. Force +0j on
        # its real axis: signed negative zero must not select the incoming root.
        gamma = cmath.sqrt(complex(radicand.real, max(0., radicand.imag)))
        self._k0 = k0
        self._epsilon = epsilon
        self._k_parallel = k_parallel
        self._gamma = gamma

    @property
    def k0(self):
        return self._k0

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def k_parallel(self):
        return self._k_parallel

    @property
    def gamma(self):
        """Decay exponent in um^-1, Re>=0 and Im>=0 for passive epsilon."""
        return self._gamma

    @property
    def kz(self):
        """Outgoing longitudinal wavenumber -i*gamma, in um^-1."""
        return -1j * self._gamma

    def _difference(self, z, z_prime):
        z = _numeric(z, 'z')
        z_prime = _numeric(z_prime, 'z_prime')
        # Convert integer inputs before subtraction to avoid integer wraparound.
        try:
            with np.errstate(over='raise', invalid='raise'):
                difference = np.asarray(z, dtype=float) - np.asarray(z_prime, dtype=float)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError('coordinate difference outside finite float64 range') from exc
        return difference

    def _exponential(self, difference):
        try:
            with np.errstate(over='raise', invalid='raise'):
                exponent = -self._gamma * np.abs(difference)
                return np.exp(exponent)
        except FloatingPointError as exc:
            raise ValueError('propagation exponent outside finite float64 range') from exc

    def __call__(self, z, z_prime):
        """Continuous G(z,z_prime), including its finite value at the source."""
        difference = self._difference(z, z_prime)
        # Divide in this order so forming 2*gamma need not overflow.
        return (0.5 / self._gamma) * self._exponential(difference)

    def derivative(self, z, z_prime, *, side=None):
        """Receiver derivative dG/dz; source limits are +1/2 left, -1/2 right.

        At z==z_prime, side='left' or 'right' is required. The default rejects
        an entire batch containing any source point rather than inventing a
        continuous derivative. side affects only exact source coincidences.
        """
        if side is not None and (not isinstance(side, str) or side not in ('left', 'right')):
            raise ValueError("side must be None, 'left' or 'right'")
        difference = self._difference(z, z_prime)
        at_source = difference == 0
        if side is None and np.any(at_source):
            raise ValueError("derivative at the source requires side='left' or 'right'")
        sign = np.sign(difference)
        if side is not None:
            sign = np.where(at_source, -1 if side == 'left' else 1, sign)
        return -0.5 * sign * self._exponential(difference)


class PoleError(ValueError):
    """Outgoing boundary system is singular or numerically unresolved near a pole."""


def _one_minus_exp(t):
    """1-exp(-t), resolving small t and allowing only propagation underflow."""
    if not (math.isfinite(t.real) and math.isfinite(t.imag)):
        raise ValueError('propagation exponent outside finite float64 range')
    with np.errstate(under='ignore'):
        return -complex(np.expm1(-t))


def _layer_element(gamma, width):
    """Exact complex endpoint DtN: outward derivatives = [[a,b],[b,a]] u.

    Only nonpositive-real exponentials occur. Small arguments use analytic
    series (including the exact gamma=0 linear element), not cutoff rejection.
    """
    t = gamma * width
    if not (math.isfinite(t.real) and math.isfinite(t.imag)):
        raise ValueError('element phase outside finite float64 range')
    if abs(t) < 1e-3:
        t2 = t*t
        a = (1+t2/3-t2*t2/45+2*t2**3/945)/width
        b = -(1-t2/6+7*t2*t2/360-31*t2**3/15120)/width
    else:
        # cmath.exp naturally allows harmless underflow without touching NumPy's
        # process/thread floating-point policy; expm1 resolves thin lossy layers.
        e = cmath.exp(-t)
        with np.errstate(under='ignore'):
            den = -complex(np.expm1(-2*t))
        a = gamma*((1+e*e)/den)
        b = -2*gamma*e/den
    if not all(math.isfinite(x.real) and math.isfinite(x.imag) for x in (a, b)):
        raise ValueError('element stiffness outside finite float64 range')
    return a, b


class LayeredGreen:
    """Complete outgoing scalar TE Green kernel for planar passive layers.

    LayeredGreen(k0, eps, thickness_um, k_parallel=0, *, origin_um=0).
    eps[0]/eps[-1] are infinite halfspaces; thickness_um describes ONLY the
    internal layers, and may be empty for a single interface. The first
    interface is origin_um. Units: um, um^-1, relative epsilon. The operator,
    time/passivity convention and source jump are those of HomogeneousGreen.

    Exact endpoint DtN elements plus outgoing halfspace Robin maps solve the
    original interface problem, not a receiver-local homogeneous approximation.
    Exact endpoint shape functions distribute an interior delta onto the mesh;
    adding its zero-endpoint element Green function recovers the source kink.
    An interface delta is a unit nodal load. Thus G'(s+)-G'(s-)=-1, including
    interface sources. Complex symmetric (not Hermitian) stiffness preserves
    lossy transpose reciprocity. Weakly attenuating oscillatory layers are
    subdivided to phase <=pi/3; layers with Re(gamma*d)>=1 need no splitting.
    Evanescent elements use decaying exponentials only, however thick.
    No growing transfer matrices or source-dependent near-duplicate nodes.

    Public configuration is scalar/tuple and getter-only; calls broadcast with
    no mutable shape/source cache. This reference implementation solves a dense
    system per coordinate pair, with at most 512 nodes (no speed claim).
    Exact INTERNAL gamma=0 is supported. Exact exterior cutoff is explicitly
    unsupported (CutoffError): a threshold radiation prescription is not
    supplied here. PoleError diagnoses singular/numerically unresolved systems
    at 32*machine-epsilon relative singular value; it does not regularize or
    clamp them. Resolved near poles retain their physical enhancement. Extreme
    dynamic ranges and sub-roundoff pole separations are not certified.
    """

    __slots__ = ('_k0', '_eps', '_thickness_um', '_k_parallel', '_origin_um',
                 '_interfaces_um', '_gamma', '_nodes')

    def __init__(self, k0, eps, thickness_um, k_parallel=0, *, origin_um=0):
        k0 = _scalar(k0, 'k0')
        kp = _scalar(k_parallel, 'k_parallel')
        origin = _scalar(origin_um, 'origin_um')
        e = _numeric(eps, 'eps', complex_allowed=True)
        d = _numeric(thickness_um, 'thickness_um')
        if e.ndim != 1 or e.size < 2:
            raise ValueError('eps requires two halfspaces and optional internal layers')
        if d.ndim != 1 or d.size != e.size-2 or np.any(d <= 0):
            raise ValueError('thickness_um requires exactly the positive internal thicknesses')
        if k0 <= 0 or kp < 0:
            raise ValueError('k0 must be positive and k_parallel nonnegative')
        if np.any(e.imag > 0):
            raise ValueError('active epsilon unsupported: Im(epsilon) must be <=0')
        k2, p2 = k0*k0, kp*kp
        if not math.isfinite(k2) or not math.isfinite(p2) or k2 == 0:
            raise ValueError('squared wavenumber outside resolved float64 range')
        epsilon = tuple(complex(v) for v in e)
        widths = tuple(float(v) for v in d)
        gamma = []
        for v in epsilon:
            rad = p2-k2*v
            if not (math.isfinite(rad.real) and math.isfinite(rad.imag)):
                raise ValueError('derived gamma squared outside finite float64 range')
            gamma.append(cmath.sqrt(complex(rad.real, max(0., rad.imag))))
        if gamma[0] == 0 or gamma[-1] == 0:
            raise CutoffError('exterior cutoff requires a threshold radiation prescription')
        edges = [origin]
        nodes = [origin]
        for width, q in zip(widths, gamma[1:-1]):
            right = edges[-1]+width
            if not math.isfinite(right) or right <= edges[-1]:
                raise ValueError('interfaces outside resolved float64 coordinate range')
            # Strongly attenuating elements have no near-sine pole even when
            # their imaginary phase is large; do not subdivide thick barriers.
            phase = 0. if q.real*width >= 1 else q.imag*width
            if not math.isfinite(phase) or phase > 500*np.pi/3:
                raise ValueError('oscillatory subdivision exceeds 512-node budget')
            count = max(1, math.ceil(phase/(np.pi/3)))
            nodes.extend(edges[-1]+width*j/count for j in range(1,count+1))
            nodes[-1] = right  # exact shared material endpoint, not rounded subdivision
            edges.append(right)
        if len(nodes)+2 > 512 or any(b <= a for a,b in zip(nodes,nodes[1:])):
            raise ValueError('mesh exceeds node budget or floating-point resolution')
        self._k0, self._k_parallel, self._origin_um = k0, kp, origin
        self._eps, self._thickness_um = epsilon, widths
        self._interfaces_um = tuple(edges)
        self._gamma, self._nodes = tuple(gamma), tuple(nodes)

    @classmethod
    def from_structure(cls, structure, k_parallel=0):
        """Snapshot a k0/eps/thickness_um structure (e.g. TEWaveguide), no mode."""
        return cls(structure.k0, structure.eps, structure.thickness_um, k_parallel,
                   origin_um=getattr(structure, 'origin_um', 0))

    @property
    def k0(self):
        return self._k0

    @property
    def k_parallel(self):
        return self._k_parallel

    @property
    def eps(self):
        return self._eps

    @property
    def thickness_um(self):
        return self._thickness_um

    @property
    def interfaces_um(self):
        return self._interfaces_um

    @property
    def origin_um(self):
        return self._origin_um

    def eps_profile(self, z):
        """Right-layer convention at interfaces; returns independent values."""
        z = _numeric(z, 'z')
        return np.asarray(self.eps)[np.searchsorted(self.interfaces_um, z, side='right')]

    def _location(self, z):
        return int(np.searchsorted(self._nodes, z, side='right'))-1

    def _shapes(self, index, z, derivative=False):
        left, right = self._nodes[index:index+2]
        width = right-left
        q = self._gamma[int(np.searchsorted(self.interfaces_um,left,side='right'))]
        x = z-left
        t = q*width
        if abs(t) < 1e-3:
            def sinhc(v):
                return 1+v*v/6+v**4/120+v**6/5040
            den = sinhc(t)
            if derivative:
                return (-cmath.cosh(q*(width-x))/(width*den),
                        cmath.cosh(q*x)/(width*den))
            return ((width-x)/width*sinhc(q*(width-x))/den,
                    x/width*sinhc(q*x)/den)
        den = _one_minus_exp(2*t)
        if derivative:
            return (-q*cmath.exp(-q*x)*(1+cmath.exp(-2*q*(width-x)))/den,
                    q*cmath.exp(-q*(width-x))*(1+cmath.exp(-2*q*x))/den)
        return (cmath.exp(-q*x)*_one_minus_exp(2*q*(width-x))/den,
                cmath.exp(-q*(width-x))*_one_minus_exp(2*q*x)/den)

    def _local(self, index, receiver, source, derivative, side):
        """Zero-endpoint element/halfline Green contribution; all exponents decay."""
        last = len(self._nodes)-1
        if index < 0 or index == last:
            lower = index < 0
            edge = self._nodes[0 if lower else -1]
            q = self._gamma[0 if lower else -1]
            x, s = abs(receiver-edge), abs(source-edge)
            distance = abs(receiver-source)
            direct = cmath.exp(-q*distance)
            image = cmath.exp(-q*(x+s))
            if not derivative:
                # Difference factored to retain precision close to the boundary.
                return direct*_one_minus_exp(q*(x+s-distance))/(2*q)
            sign = ((-1 if side == 'left' else 1) if receiver == source
                    else (-1 if receiver < source else 1))
            direction = -1 if lower else 1
            return (-sign*direct+direction*image)/2
        left,right = self._nodes[index:index+2]
        width = right-left
        x,s = receiver-left, source-left
        q = self._gamma[int(np.searchsorted(self.interfaces_um,left,side='right'))]
        before = receiver < source or (receiver == source and side == 'left')
        a,b = min(x,s), width-max(x,s)
        t = q*width
        if abs(t) < 1e-3:
            def sinhc(v):
                return 1+v*v/6+v**4/120+v**6/5040
            den = sinhc(t)
            if derivative:
                return ((width-s)/width*cmath.cosh(q*x)*sinhc(q*(width-s))/den if before else
                        -s/width*sinhc(q*s)*cmath.cosh(q*(width-x))/den)
            return a*b/width*sinhc(q*a)*sinhc(q*b)/den
        den = _one_minus_exp(2*t)
        e = cmath.exp(-q*abs(x-s))
        if derivative:
            return (.5*(1+cmath.exp(-2*q*x))*_one_minus_exp(2*q*(width-s))*e/den if before else
                    -.5*_one_minus_exp(2*q*s)*(1+cmath.exp(-2*q*(width-x)))*e/den)
        return _one_minus_exp(2*q*a)*_one_minus_exp(2*q*b)*e/(2*q*den)

    def _pair(self, receiver, source, derivative, side):
        n = len(self._nodes)
        matrix = np.zeros((n,n), complex)
        for i in range(n-1):
            left,right = self._nodes[i:i+2]
            layer = int(np.searchsorted(self.interfaces_um,left,side='right'))
            a,b = _layer_element(self._gamma[layer],right-left)
            matrix[i,i] += a
            matrix[i+1,i+1] += a
            matrix[i,i+1] = matrix[i+1,i] = b
        matrix[0,0] += self._gamma[0]
        matrix[-1,-1] += self._gamma[-1]
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[-1] <= 32*np.finfo(float).eps*singular[0]:
            raise PoleError('outgoing Green system singular or numerically unresolved near a pole')
        source_index = self._location(source)
        receiver_index = self._location(receiver)
        rhs = np.zeros(n,complex)
        if source_index < 0 or source_index == n-1:
            lower = source_index < 0
            j = 0 if lower else -1
            rhs[j] = cmath.exp(-self._gamma[j]*abs(source-self._nodes[j]))
        else:
            rhs[source_index:source_index+2] = self._shapes(source_index,source)
        try:
            value = np.linalg.solve(matrix,rhs)
        except np.linalg.LinAlgError as exc:
            raise PoleError('outgoing Green system singular at a pole') from exc
        if derivative and receiver == source and side == 'left' and receiver in self._nodes:
            receiver_index -= 1
        if receiver_index < 0 or receiver_index == n-1:
            lower = receiver_index < 0
            j = 0 if lower else -1
            q = self._gamma[j]
            result = complex(value[j])*cmath.exp(-q*abs(receiver-self._nodes[j]))
            if derivative:
                result *= q if lower else -q
        else:
            a,b = self._shapes(receiver_index,receiver,derivative)
            result = a*complex(value[receiver_index])+b*complex(value[receiver_index+1])
        if receiver_index == source_index:
            result += self._local(receiver_index,receiver,source,derivative,side)
        return result

    def _evaluate(self, z, z_prime, derivative=False, side=None):
        if side is not None and (not isinstance(side,str) or side not in ('left','right')):
            raise ValueError("side must be None, 'left' or 'right'")
        z,s = np.broadcast_arrays(_numeric(z,'z'), _numeric(z_prime,'z_prime'))
        if derivative and side is None and np.any(z == s):
            raise ValueError("derivative at the source requires side='left' or 'right'")
        result = np.empty(z.shape,complex)
        for index in np.ndindex(z.shape):
            result[index] = self._pair(float(z[index]),float(s[index]),derivative,side)
        return result.item() if result.ndim == 0 else result

    def __call__(self, z, z_prime):
        """Continuous G, including interface sources; broadcasts z and z_prime."""
        return self._evaluate(z,z_prime)

    def derivative(self, z, z_prime, *, side=None):
        """Receiver derivative; side is required only at exact source coincidence.

        Interfaces use the right layer, with continuous derivative except for
        an interface source. side='left'/'right' then selects the source limit.
        """
        return self._evaluate(z,z_prime,True,side)
