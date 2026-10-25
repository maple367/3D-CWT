"""Lossless square-lattice TE reference coupling, all C terms in um^-1.

V=(Rx,Sx,Ry,Sy), harmonics=(+x,-x,+y,-y), fields=(y,y,x,x).
Liang thesis (3.27)-(3.35), extended with LOCAL receiver/source xi in
multiple PC layers. No averaging of xi before applying the Green operator.
No legacy imports, output files, material-loss truncation or hidden redesign.
"""
from dataclasses import dataclass
import numpy as np
from .geometry import CircleHole, RightTriangleHole, SampledPermittivity, _integer
from .fourier import shape_fourier, fft_fourier
from .greens import LayeredGreen, PreparedLayeredGreen
from .quadrature import mode_overlap, green_overlap
from .waveguide import TEMode, _positive_scalar


def _pack(**arrays):
    return tuple((name,a.shape,a.dtype.str,a.tobytes())
                 for name,value in arrays.items() for a in [np.asarray(value)])


class _Snapshots:
    def __getattr__(self,name):
        for key,shape,dtype,data in self._arrays:
            if name == key:
                return np.ndarray(shape,dtype=dtype,buffer=data)
        raise AttributeError(name)


@dataclass(frozen=True)
class CouplingResult(_Snapshots):
    """Independent bytes-backed array headers on EVERY read.

    C1D=kappa feedback; Crad=zeta; C2D=chi including -k0²/(2 beta0).
    C2D_green and C2D_local separate mu and negative-nu contributions.
    mu/nu have axes (auxiliary, receiver layer, source basic wave), and
    varsigma has (auxiliary, receiver layer, component x/y, source wave).
    They are dimensionless receiver-layer integrated fields, NOT averaged xi.
    Summing the receiver-layer axis gives thesis (3.28)/(3.29)/(3.27).
    I,J,K are layer overlaps; K radial axis corresponds to radial_orders
    (zero is radiation). xi is the Fourier table with indices xi_orders.
    No a-priori quadrature error estimate: order/cutoff are explicit metadata.
    """
    _arrays: tuple
    period_um: float
    beta0: float
    k0: float
    relative_detuning: float
    cutoff: int
    quadrature_order: int
    pc_layers: tuple
    radial_orders: tuple
    all_aux_evanescent: bool
    units: str = 'um^-1'
    fourier_order: int = 64
    backend: str = 'reference'
    sampling_shapes: tuple = ()
    # Immutable canonical input snapshot; empty for manually constructed results.
    # Not an authentication signature against deliberate dataclass forgery.
    provenance: str = ''


def build_coupling(mode, period_um, pc_layers, *, cutoff=1,
                   quadrature_order=12, fourier_order=64, beta_rtol=1e-8,
                   backend='reference'):
    """Build physical reference C using a normalized TEMode.

    pc_layers maps guide epsilon indices 1..len(eps)-2 to a CircleHole,
    RightTriangleHole or SampledPermittivity. Others have xi=0; empty mapping
    is legal and gives C=0. Each geometry's mean MUST match guide epsilon.
    Scalar period is explicitly square and must match mode.beta within
    beta_rtol (default 1e-8 relative). A relaxed tolerance records detuning
    but does not add a detuning diagonal or alter period/k0. This is the
    near-Bragg coupling operator, not a self-consistent frequency solve.
    Quadrature is fixed-order per material interval. Kernels/overlaps reused
    by m²+n², never recomputed per matrix entry. Scalar TE/nonmagnetic only.
    Propagating auxiliaries allowed; C2D Hermiticity then not promised.
    backend='reference' is the unchanged dense per-pair regression anchor;
    'prepared' uses one pole check/LU per radial kernel and batched source RHS,
    with identical Fourier/quadrature orders and same-layer Duffy integration.
    fourier_order and sampled grid shapes are retained in result metadata.
    """
    if not isinstance(mode,TEMode):
        raise TypeError('mode must be a normalized TEMode')
    if backend not in ('reference', 'prepared'):
        raise ValueError("backend must be 'reference' or 'prepared'")
    a = _positive_scalar(period_um,'period_um')
    D = _integer(cutoff,'cutoff')
    Q = _integer(quadrature_order,'quadrature_order')
    F = _integer(fourier_order,'fourier_order')
    if isinstance(beta_rtol,(bool,np.bool_)) or not np.isscalar(beta_rtol) or not np.isfinite(beta_rtol) or beta_rtol < 0:
        raise ValueError('beta_rtol must be finite and nonnegative')
    b = 2*np.pi/a
    if not np.isfinite(b) or b <= 0 or not np.isfinite(b*b) or b*b == 0:
        raise ValueError('beta0 or beta0 squared outside resolved finite range')
    detuning = (mode.beta-b)/b
    if abs(detuning) > beta_rtol:
        raise ValueError('mode.beta does not match beta0=2*pi/period_um within beta_rtol')
    guide = mode.guide
    if guide.period_um is not None and not np.isclose(guide.period_um,a,rtol=1e-12,atol=0):
        raise ValueError('period_um differs from guide fixed period')
    layers = dict(pc_layers)
    for l,h in layers.items():
        if isinstance(l,(bool,np.bool_)) or not isinstance(l,(int,np.integer)) or not 1 <= l <= len(guide.eps)-2:
            raise ValueError('PC layer indices must be finite internal epsilon indices')
        if not isinstance(h,(CircleHole,RightTriangleHole,SampledPermittivity)):
            raise TypeError('Unsupported PC geometry')
        if not np.allclose(h.period_um,(a,a),rtol=1e-12,atol=0):
            raise ValueError('geometry must have the specified square period')
        material = h.values if isinstance(h,SampledPermittivity) else np.array([h.eps_background,h.eps_hole])
        if np.any(material.imag != 0) or np.any(material.real <= 0):
            raise ValueError('PC materials must be positive lossless epsilon')
        mean = material.mean() if isinstance(h,SampledPermittivity) else h.mean_epsilon
        if not np.isclose(mean,guide.eps[l],rtol=1e-11,atol=1e-12):
            raise ValueError('geometry mean epsilon does not match averaged waveguide')
    # Lazy import avoids a module-initialization cycle with the optional cache.
    # Snapshot actual inputs, not a caller-supplied key or inferred Fourier table.
    from .cache import _document, _json
    provenance = _json(_document(mode, a, layers, dict(
        cutoff=cutoff, quadrature_order=quadrature_order, fourier_order=fourier_order,
        beta_rtol=beta_rtol, backend=backend)))
    ids = tuple(sorted(layers))
    basic = np.array([(1,0),(-1,0),(0,1),(0,-1)])
    aux = [(m,n) for m in range(-D,D+1) for n in range(-D,D+1) if m*m+n*n > 1]
    radial = (0,*sorted({m*m+n*n for m,n in aux}))
    orders = np.array([(m,n) for m in range(-D-1,D+2) for n in range(-D-1,D+2)],dtype=int)
    lookup = {tuple(o):i for i,o in enumerate(orders)}
    xi = np.empty((len(ids),len(orders)),complex)
    for i,l in enumerate(ids):
        h = layers[l]
        xi[i] = (fft_fourier(h,orders) if isinstance(h,SampledPermittivity)
                 else shape_fourier(h,orders,quadrature_order=F))
        xi[i,lookup[(0,0)]] = 0 # mean is already in the waveguide
    def coeff(order):
        return xi[:,lookup[tuple(order)]]
    edges = guide.interfaces_um
    bounds = [(edges[l-1],edges[l]) for l in ids]
    I = np.array([mode_overlap(mode,t,Q) for t in bounds])
    J = I/np.array([guide.eps[l] for l in ids])
    K = np.empty((len(radial),len(ids),len(ids)),complex)
    for index,r2 in enumerate(radial):
        kernel_type = LayeredGreen if backend == 'reference' else PreparedLayeredGreen
        kernel = kernel_type.from_structure(guide,b*np.sqrt(r2))
        for i,left in enumerate(bounds):
            for j,right in enumerate(bounds):
                K[index,i,j] = green_overlap(mode,left,right,kernel,Q)
    f = -guide.k0**2/(2*b)
    c1,cr,cg,cl = [np.zeros((4,4),complex) for _ in range(4)]
    for i,p in enumerate(basic):
        for j,r in enumerate(basic):
            if j == (i^1):
                c1[i,j] = f*np.dot(coeff(p-r),I)
            if i//2 == j//2:
                cr[i,j] = f*guide.k0**2*(coeff(p)@K[0]@coeff(-r))
    # Eminus=k0² G A xi Theta; Eplus=-B xi Theta/epsilon.
    # Ex=(n Eminus+m Eplus)/r²; Ey=(-m Eminus+n Eplus)/r².
    mus, nus, fields = [], [], []
    for m,n in aux:
        r2 = m*m+n*n
        source = np.stack([coeff(np.array([m,n])-r) for r in basic],axis=1)
        receiver = np.stack([coeff(p-np.array([m,n])) for p in basic])
        mu = guide.k0**2*(K[radial.index(r2)]@source)
        nu = -J[:,None]*source # retain (3.29)'s minus sign
        A,B = np.array([-m,-m,n,n]),np.array([n,n,m,m])
        transverse,longitudinal = np.array([-m,-m,n,n]),np.array([n,n,m,m])
        mus.append(mu)
        nus.append(nu)
        # varsigma is integral of Theta* E_(x,y) over EACH receiver layer,
        # before multiplying that layer's receiver xi; axes (layer, x/y, j).
        fields.append(np.stack([(n*mu*A+m*nu*B)/r2,
                                (-m*mu*A+n*nu*B)/r2],axis=1))
        cg += f*(receiver@mu)*transverse[:,None]*A[None,:]/r2
        cl += f*(receiver@nu)*longitudinal[:,None]*B[None,:]/r2
    c2 = cg+cl
    total = c1+cr+c2
    if not np.isfinite(total).all():
        raise ValueError('coupling outside finite numeric range')
    return CouplingResult(_pack(C1D=c1,Crad=cr,C2D=c2,C2D_green=cg,C2D_local=cl,
                                C=total,I=I,J=J,K=K,xi=xi,xi_orders=orders,
                                aux_orders=np.array(aux,dtype=int),
                                mu=np.array(mus),nu=np.array(nus),varsigma=np.array(fields)),
                          a,b,guide.k0,float(detuning),D,Q,ids,radial,
                          bool(all(b*b*r > guide.k0**2*max(guide.eps) for r in radial[1:])),
                          fourier_order=F, backend=backend, provenance=provenance,
                          sampling_shapes=tuple((l,layers[l].values.shape) for l in ids
                                                if isinstance(layers[l],SampledPermittivity)))
