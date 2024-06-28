import numpy as np
from scipy.integrate import dblquad, quad, trapezoid, simpson, romb, qmc_quad

def kappa_func(gamma_phc, xi_array):
    return gamma_phc*xi_array

def _dblquad_complex(self, func, a, b, gfun, hfun, args=(), **kwargs):
    """Double quad integration. Suitable for real number space integral routine.
    The function should return a complex number and abserr.
    
    Parameters
    ----------
    func : callable
        The function to be integrated. func(x, y, *args, **kwargs)
    a, b : float
        The limits of integration in x.
    gfun, hfun : callable
        The limits of integration in y.
    args : tuple, optional
        Extra arguments to pass to func.
    **kwargs : dict, optional
        Extra keyword arguments to pass to func.

    Returns
    -------
    out : tuple
        The result and absolute error of the double integration."""
    def real_func(y, x, *args):
        return np.real(func(x, y, *args))
    def imag_func(y, x, *args):
        return np.imag(func(x, y, *args))
    real_integral = dblquad(real_func, a, b, gfun, hfun, args=args, **kwargs)
    imag_integral = dblquad(imag_func, a, b, gfun, hfun, args=args, **kwargs)
    return real_integral[0] + 1j*imag_integral[0], real_integral[1] + 1j*imag_integral[1]

def _dbltrapezoid(self, func, XX, YY, args=()):
    """Double trapezoid integration. Only siutable for rectangular zone."""
    ZZ = func(XX, YY, *args)
    x = XX[:, 0]
    return trapezoid(trapezoid(ZZ, YY, axis=1), x, axis=0)

def _dblsimpson(self, func, XX, YY, args=()):
    """Double Simpson integration. Only siutable for rectangular zone."""
    ZZ = func(XX, YY, *args)
    x = XX[:, 0]
    return simpson(simpson(ZZ, YY, axis=1), x, axis=0)

def _dblromb(self, func, XX, YY, dx, dy, args=()):
    """Double romberg integration. Only siutable for rectangular zone. The sample number should be 2^n+1."""
    ZZ = func(XX, YY, *args)
    return romb(romb(ZZ, dy, axis=1), dx, axis=0)

def _dblqmc_quad(self, func, a, b, c, d, args=(), **kwargs):
    """Double quasi-Monte Carlo integration. Suitable for real number space integral routine."""
    def real_func(coor):
        x, y = coor
        return np.real(func(x, y, *args))
    def imag_func(coor):
        x, y = coor
        return np.imag(func(x, y, *args))
    real_integral = qmc_quad(real_func, np.array([a, c]), np.array([b, d]), **kwargs)
    imag_integral = qmc_quad(imag_func, np.array([a, c]), np.array([b, d]), **kwargs)
    return real_integral[0] + 1j*imag_integral[0], real_integral[1] + 1j*imag_integral[1]

def _quad_complex(self, func, a, b, args=(), **kwargs):
    """Quad integration. Suitable for real number space integral routine.
    The function should return a complex number and abserr.
    
    Parameters
    ----------
    func : callable
        The function to be integrated. func(x, *args, **kwargs)
    a, b : float
        The limits of integration in x.
    args : tuple, optional
        Extra arguments to pass to func.
    **kwargs : dict, optional
        Extra keyword arguments to pass to func.

    Returns
    -------
    out : tuple
        The result and absolute error of the double integration."""
    def real_func(y, x, *args):
        return np.real(func(x, y, *args))
    def imag_func(y, x, *args):
        return np.imag(func(x, y, *args))
    real_integral = quad(real_func, a, b, args=args, **kwargs)
    imag_integral = quad(imag_func, a, b, args=args, **kwargs)
    return real_integral[0] + 1j*imag_integral[0], real_integral[1] + 1j*imag_integral[1]