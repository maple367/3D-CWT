import numpy as np
from scipy.integrate import dblquad, quad, trapezoid, simpson, romb, qmc_quad

class integral_method():
    """
    accurency_factor : int [default: 3]
        For quad, epsrel = 10**(-2*accurency_factor)
        For trapz and etc., reslution = int(2**(accurency_factor+5)+1)
    The integral method class. The class contains the integral method for complex number space.
    The class contains the following methods:

    Parameters
    ----------
    accurency_factor : int [default: 3]
        The accurency factor for the integral method.
    method : str [default: 'dblquad']
        The integral method. {'dblquad', 'dbltrapezoid', 'dblsimpson', 'dblromb', 'dblqmc_quad', 'quad'}

    Returns
    -------
    out : Class
        call : return the integral method.
    """

    def __init__(self, accurency_factor=3, method='dblquad'):
        self.accurency_factor = accurency_factor
        self.method = method

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
        real_integral = dblquad(lambda y,x: func(x,y,*args).real, a, b, gfun, hfun, epsrel=self.epsrel, **kwargs)
        imag_integral = dblquad(lambda y,x: func(x,y,*args).imag, a, b, gfun, hfun, epsrel=self.epsrel, **kwargs)
        print(f'abserr: {real_integral[1] + 1j*imag_integral[1]}')
        return real_integral[0] + 1j*imag_integral[0]

    def _dbltrapezoid(self, func, a, b, c, d, args=(), **kwargs):
        """Double trapezoid integration. Only siutable for rectangular zone."""
        XX, YY = np.meshgrid(np.linspace(a, b, self.reslution), np.linspace(c, d, self.reslution), indexing='ij')
        ZZ = func(XX, YY, *args)
        x = XX[:, 0]
        return trapezoid(trapezoid(ZZ, YY, axis=1, **kwargs), x, axis=0, **kwargs)

    def _dblsimpson(self, func, a, b, c, d, args=(), **kwargs):
        """Double Simpson integration. Only siutable for rectangular zone."""
        XX, YY = np.meshgrid(np.linspace(a, b, self.reslution), np.linspace(c, d, self.reslution), indexing='ij')
        ZZ = func(XX, YY, *args)
        x = XX[:, 0]
        return simpson(simpson(ZZ, YY, axis=1, **kwargs), x, axis=0, **kwargs)

    def _dblromb(self, func, a, b, c, d, args=(), **kwargs):
        """Double romberg integration. Only siutable for rectangular zone. The sample number should be 2^n+1."""
        x, dx = np.linspace(a, b, self.reslution, retstep=True)
        y, dy = np.linspace(c, d, self.reslution, retstep=True)
        XX, YY = np.meshgrid(x, y, indexing='ij')
        ZZ = func(XX, YY, *args)
        return romb(romb(ZZ, dy, axis=1, **kwargs), dx, axis=0, **kwargs)

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
        def real_func(x, *args):
            return np.real(func(x, *args))
        def imag_func(x, *args):
            return np.imag(func(x, *args))
        real_integral = quad(real_func, a, b, args=args, epsrel=self.epsrel, **kwargs)
        imag_integral = quad(imag_func, a, b, args=args, epsrel=self.epsrel, **kwargs)
        print(f'abserr: {real_integral[1] + 1j*imag_integral[1]}')
        return real_integral[0] + 1j*imag_integral[0]
    
    def __getattr__(self, name):
        if name == 'epsrel':
            return 10**(-2*self.accurency_factor)
        if name == 'reslution':
            return int(2**(self.accurency_factor+5)+1)
        
    def __call__(self):
        if self.method == 'dblquad':
            return self._dblquad_complex
        if self.method == 'dbltrapezoid':
            return self._dbltrapezoid
        if self.method == 'dblsimpson':
            return self._dblsimpson
        if self.method == 'dblromb':
            return self._dblromb
        if self.method == 'dblqmc_quad':
            return self._dblqmc_quad
        if self.method == 'quad':
            return self._quad_complex