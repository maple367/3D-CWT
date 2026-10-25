"""Positive-exponent Fourier coefficients xi(m,n), normalized by cell area.

Reconstruction uses sum xi(m,n)*exp(-2*pi*i*(m*x/ax+n*y/ay)). Orders
have shape (...,2), last axis (m,n); results have shape (...). No global
or object spectrum cache is used. Pure NumPy/SciPy; no legacy backend.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import j1

from .geometry import (CircleHole, RightTriangleHole, SampledPermittivity,
                       _epsilon, _integer, _pair)


def _orders(orders):
    a = np.asarray(orders)
    if a.ndim < 1 or a.shape[-1] != 2:
        raise ValueError('orders must have shape (...,2), last axis (m,n)')
    if a.dtype.kind not in 'iu':
        raise TypeError('Harmonic orders must be integers (not bool or floating point)')
    if np.any(a > np.iinfo(np.int64).max):
        raise ValueError('Order outside signed int64 range')
    return a.astype(np.int64)


def _gauss(order):
    q = _integer(order, 'quadrature_order')
    nodes, weights = leggauss(q)
    return (nodes+1)/2, weights/2


def _weighted_coefficients(x, y, values, weights, period, orders):
    x, y, values, weights = np.broadcast_arrays(x, y, values, weights)
    # Loop over harmonics avoids allocating orders x all quadrature nodes.
    result = [np.sum(values*weights*np.exp(2j*np.pi*(m*x/period[0]+n*y/period[1])))
              for m,n in orders.reshape(-1,2)]
    return np.asarray(result, dtype=complex).reshape(orders.shape[:-1])


def integrate_fourier(epsilon, period_um, orders, *, quadrature_order=64):
    """Reference tensor Gauss-Legendre integration over a rectangular cell.

    Callable epsilon(x_um,y_um) must broadcast; complex values are allowed.
    quadrature_order nodes per axis; no adaptive tolerance or error estimate
    is promised. Compare increasing orders to establish accuracy. Reliable for
    smooth resolved functions; NOT a high-accuracy shortcut for discontinuous
    arbitrary masks. For holes use integrate_hole_fourier; otherwise split the
    domain or establish sampling/integration convergence independently.
    """
    p, o = _pair(period_um, 'period_um', positive=True), _orders(orders)
    t,w = _gauss(quadrature_order)
    x,y = t[None,:]*p[0], t[:,None]*p[1]
    values = np.broadcast_to(_epsilon(epsilon(x,y)), (len(t),len(t)))
    return _weighted_coefficients(x,y,values,w[:,None]*w[None,:],p,o)


def integrate_hole_fourier(hole, orders, *, quadrature_order=64):
    """Integrate only the smooth inclusion contrast, with exact background.

    Circle: polar Gauss quadrature, including r Jacobian. Right triangle:
    Duffy map (u,v) -> vertex+(lx*u, ly*(1-u)*v), Jacobian lx*ly*(1-u).
    Fixed order per axis; convergence must be checked for large harmonics.
    No discontinuous full-cell mask quadrature is used.
    """
    if not isinstance(hole, (CircleHole,RightTriangleHole)):
        raise TypeError('Expected CircleHole or RightTriangleHole')
    o = _orders(orders)
    t,w = _gauss(quadrature_order)
    u,v = t[None,:],t[:,None]
    weights = w[:,None]*w[None,:]
    if isinstance(hole,CircleHole):
        r, theta = hole.radius_um*u, 2*np.pi*v
        x = hole.center_um[0]+r*np.cos(theta)
        y = hole.center_um[1]+r*np.sin(theta)
        jacobian = 2*np.pi*hole.radius_um*r
    else:
        lx,ly = hole.legs_um
        vertex = hole.vertices_um[0]
        x = vertex[0]+lx*u
        y = vertex[1]+ly*(1-u)*v
        jacobian = lx*ly*(1-u)
    contrast = (hole.eps_hole-hole.eps_background)/np.prod(hole.period_um)
    result = _weighted_coefficients(x,y,contrast,weights*jacobian,hole.period_um,o)
    return result + hole.eps_background*np.all(o==0,axis=-1)


def shape_fourier(hole, orders, *, quadrature_order=64):
    """Circle analytic Bessel coefficients; right triangle Duffy reference.

    Circle zero order uses the exact limit 2*J1(z)/z -> 1. Translation is
    exp(+i G dot center), valid for complex materials without conjugation.
    quadrature_order applies only to triangles (see integrate_hole_fourier).
    """
    o = _orders(orders)
    if isinstance(hole,RightTriangleHole):
        return integrate_hole_fourier(hole,o,quadrature_order=quadrature_order)
    if not isinstance(hole,CircleHole):
        raise TypeError('Expected CircleHole or RightTriangleHole')
    g = 2*np.pi*o/np.array(hole.period_um)
    z = np.hypot(g[...,0],g[...,1])*hole.radius_um
    form = np.ones_like(z)
    np.divide(2*j1(z),z,out=form,where=z!=0)
    phase = np.exp(1j*np.sum(g*np.array(hole.center_um),axis=-1))
    return ((hole.eps_hole-hole.eps_background)*hole.fill_fraction*form*phase
            + hole.eps_background*np.all(o==0,axis=-1))


def fft_fourier(sampled, orders):
    """Batch positive-exponent DISCRETE coefficients from midpoint samples.

    ifft2 supplies the positive sign AND 1/(Nx*Ny) normalization. Indices
    address spectrum[n % Ny,m % Nx]; exp(i*pi*(m/Nx+n/Ny)) shifts the basis
    from index-origin samples to cell centers. No conjugation of epsilon.

    Unique orders only, in canonical -N//2 <= k <= (N-1)//2 (where the lower
    bound means -(N//2)). For even N the negative Nyquist representative is
    accepted, the positive one rejected. Out-of-band/aliased requests raise;
    this function never resamples, enlarges the grid, or reuses a stale cache.
    """
    if not isinstance(sampled,SampledPermittivity):
        raise TypeError('Expected SampledPermittivity')
    o = _orders(orders)
    ny,nx = sampled.values.shape
    flat = o.reshape(-1,2)
    if (np.any(flat < [- (nx//2),- (ny//2)]) or
            np.any(flat > [(nx-1)//2,(ny-1)//2])):
        raise ValueError('Orders outside canonical Nyquist interval')
    if len(np.unique(flat,axis=0)) != len(flat):
        raise ValueError('Duplicate/aliased harmonic orders')
    spectrum = np.fft.ifft2(sampled.values)
    m,n = o[...,0],o[...,1]
    return spectrum[n % ny,m % nx]*np.exp(1j*np.pi*(m/nx+n/ny))
