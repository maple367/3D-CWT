"""Interface-partitioned fixed-order reference quadrature (um).

Green's derivative has a source cusp: on a same-layer square integrate BOTH
smooth triangles, never a tensor rule across the diagonal. Fixed order is not
an error certificate; compare increasing orders independently.
"""
import numpy as np
from .geometry import _integer


def interval_rule(bounds, order):
    """Gauss nodes/weights strictly inside a finite positive interval."""
    n = _integer(order, 'quadrature_order')
    a,b = bounds
    if not np.isfinite([a,b]).all() or b <= a:
        raise ValueError('bounds must be a finite increasing pair')
    x,w = np.polynomial.legendre.leggauss(n)
    return a+(x+1)*(b-a)/2, w*(b-a)/2


def mode_overlap(mode, bounds, order):
    z,w = interval_rule(bounds,order)
    return float(np.dot(w,abs(mode.theta(z))**2))


def green_overlap(mode, receiver_bounds, source_bounds, kernel, order):
    """Integral Theta*(z) G(z,s) Theta(s) dz ds, units um^2.

    Intervals must be identical or nonoverlapping, each within one material.
    Duffy coordinates z=a+L*u, s=a+L*u*v, Jacobian L^2*u,
    plus the reflected triangle, resolve the same-interval source cusp.
    """
    z,w = interval_rule(receiver_bounds,order)
    s,v = interval_rule(source_bounds,order)
    if tuple(receiver_bounds) != tuple(source_bounds):
        if max(receiver_bounds[0],source_bounds[0]) < min(receiver_bounds[1],source_bounds[1]):
            raise ValueError('intervals must be identical or nonoverlapping')
        return np.sum(w[:,None]*v[None,:]*mode.theta(z)[:,None].conj()*
                      kernel(z[:,None],s[None,:])*mode.theta(s)[None,:])
    u,w = interval_rule((0.,1.),order)
    a,b = receiver_bounds
    x = a+(b-a)*u[:,None]
    y = a+(b-a)*u[:,None]*u[None,:]
    weights = (b-a)**2*u[:,None]*w[:,None]*w[None,:]
    return np.sum(weights*(mode.theta(x).conj()*kernel(x,y)*mode.theta(y)+
                           mode.theta(y).conj()*kernel(y,x)*mode.theta(x)))
