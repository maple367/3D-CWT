import numpy as np
from ._collection import integral_method
from model.rect_lattice import eps_userdefine

_dblquad_complex = integral_method(3)._dblquad_complex

class xi_calculator_DFT():
    """
    Calculate the Fourier coefficients of the dielectric constant distribution.
    Use the Discrete Fourier Transform (DFT) to calculate the Fourier coefficients.

    Parameters
    ----------
    eps_func : class eps_userdefine
        The class contains dielectric constant distribution in 2D plane and cell size.
    resolution : int
        The sampling frequency or integral step frequency.

    Returns
    -------
    out : class
        The Fourier coefficients.
    """
    def __init__(self, eps_func:eps_userdefine, resolution:int):
        import warnings
        warnings.warn("The DFT method is deprecated.", DeprecationWarning)
        cell_size_x = eps_func.cell_size_x
        cell_size_y = eps_func.cell_size_y
        x_mesh = np.linspace(0, cell_size_x, resolution)
        y_mesh = np.linspace(0, cell_size_y, resolution)
        X, Y = np.meshgrid(x_mesh, y_mesh)
        eps_array = eps_func(X, Y)
        self.xi_array = np.fft.fft2(eps_array)/(resolution*resolution)

    def __getitem__(self, index):
        return self.xi_array[index]
    
    def __mul__(self, other):
        return  self.xi_array*other
    
    def __rmul__(self, other):
        return  other*self.xi_array
    
    def get_raw_array(self):
        return self.xi_array


# not vectorized
class xi_calculator():
    """
    Calculate the Fourier coefficients of the dielectric constant distribution.

    Parameters
    ----------
    eps_func : class eps_userdefine
        The class contains dielectric constant distribution in 2D plane and cell size.
    method : str, {'dbltrapezoid', 'dblsimpson', 'dblquad', 'dblromb', 'dblqmc_quad'}
        The size of the cell in the y direction.
    **kwargs : keyword arguments, refer to example.py for details.

    Returns
    -------
    out : class
        The Fourier coefficients.
    """
    def __init__(self, eps_func:eps_userdefine, method='dblquad', **kwargs):
        self.eps_func = eps_func
        self.eps_type = self.eps_func.eps_type
        self.method = method
        self.kwargs = kwargs
        self.cell_size_x = eps_func.cell_size_x
        self.cell_size_y = eps_func.cell_size_y
        self.beta_0_x = 2*np.pi/self.cell_size_x
        self.beta_0_y = 2*np.pi/self.cell_size_y
        self.xi_array = {}
    
    def _integrated_func(self, x, y, m:int, n:int):
        return self.eps_func(x, y)*np.exp(1j*(self.beta_0_x*m*x+self.beta_0_y*n*y))

    def __getitem__(self, index):
        try:
            return self.xi_array[index]
        except:
            self.xi_array[index] = self._cal(index)
            return self.xi_array[index]
    
    def _cal(self, index:tuple[int, int]):
        if self.eps_type == 'circle':
            self._xi = self._cal_circle(index) # maybe not need to assign?
        else:
            self._xi = self._cal_general(index)
        return self._xi
    
    def _cal_circle(self, index:tuple[int, int]):
        """Calculate the Fourier coefficients of the dielectric constant distribution for circle eps_func. Circle has discontinuity, so the integration should be separated into 3 zones."""
        print(index)
        m, n = index
        if self.method == 'dblquad':
            integral_func = integral_method(3, method=self.method)()
            def boundary_yb(x):
                x = self.eps_func._x(x)
                if x < self.cell_size_x/2-self.eps_func.r or x > self.cell_size_x/2+self.eps_func.r:
                    return self.cell_size_y/2
                else:
                    return self.cell_size_y/2-np.sqrt(self.eps_func.r**2-(x-self.cell_size_x/2)**2)
            def boundary_yu(x):
                x = self.eps_func._x(x)
                if x < self.cell_size_x/2-self.eps_func.r or x > self.cell_size_x/2+self.eps_func.r:
                    return self.cell_size_y/2
                else:
                    return self.cell_size_y/2+np.sqrt(self.eps_func.r**2-(x-self.cell_size_x/2)**2)
            zone1 = integral_func(self._integrated_func, 0, self.cell_size_x, 0, boundary_yb, args=(m, n), **self.kwargs)
            zone2 = integral_func(self._integrated_func, 0, self.cell_size_x, boundary_yb, boundary_yu, args=(m, n), **self.kwargs)
            zone3 = integral_func(self._integrated_func, 0, self.cell_size_x, boundary_yu, self.cell_size_y, args=(m, n), **self.kwargs)
            zonels = [zone1, zone2, zone3]
            self._xi = np.sum([zone[0] for zone in zonels])
            self._abserr = np.sum([np.abs(np.real(zone[1]))+1j*np.abs(np.imag(zone[1])) for zone in zonels])
        elif self.method in ['dbltrapezoid', 'dblsimpson', 'dblromb', 'dblqmc_quad']:
            print(f'{self.method} not supported for circle eps_func now. Use userdefine instead.')
            self.eps_type = 'userdefine'
            return self._cal(index)
        else:
            raise ValueError('Method not supported.')
        self._xi = self._xi/(self.cell_size_x*self.cell_size_y)
        return self._xi
    
    def _cal_general(self, index:tuple[int, int]):
        m, n = index
        if self.method == 'dbltrapezoid':
            integral_func = integral_method(3, method=self.method)()
            x_mesh = np.linspace(0, self.cell_size_x, num=self.kwargs['resolution'])
            y_mesh = np.linspace(0, self.cell_size_y, num=self.kwargs['resolution'])
            XX, YY = np.meshgrid(x_mesh, y_mesh, indexing='ij')
            self._xi = integral_func(self._integrated_func, XX, YY, args=(m, n))
        elif self.method == 'dblsimpson':
            integral_func = integral_method(3, method=self.method)()
            x_mesh = np.linspace(0, self.cell_size_x, num=self.kwargs['resolution'])
            y_mesh = np.linspace(0, self.cell_size_y, num=self.kwargs['resolution'])
            XX, YY = np.meshgrid(x_mesh, y_mesh, indexing='ij')
            self._xi = integral_func(self._integrated_func, XX, YY, args=(m, n))
        elif self.method == 'dblquad':
            integral_func = integral_method(3, method=self.method)()
            self._xi, self._abserr = integral_func(self._integrated_func, 0, self.cell_size_x, 0, self.cell_size_y, args=(m, n), **self.kwargs)
        elif self.method == 'dblromb':
            integral_func = integral_method(3, method=self.method)()
            x_mesh = np.linspace(0, self.cell_size_x, num=self.kwargs['resolution'], retstep=True)
            y_mesh = np.linspace(0, self.cell_size_y, num=self.kwargs['resolution'], retstep=True)
            XX, YY = np.meshgrid(x_mesh[0], y_mesh[0], indexing='ij')
            self._xi = integral_func(self._integrated_func, XX, YY, x_mesh[1], y_mesh[1], args=(m, n))
        elif self.method == 'dblqmc_quad':
            integral_func = integral_method(3, method=self.method)()
            self._xi, self._abserr = integral_func(self._integrated_func, 0, self.cell_size_x, 0, self.cell_size_y, args=(m, n), **self.kwargs)
        else:
            raise ValueError('Method not supported.')
        self._xi = self._xi/(self.cell_size_x*self.cell_size_y)
        return self._xi

    def get_raw_array(self):
        return self.xi_array


# not vectorized
class Array_calculator():
    """
    Calculate the coefficients array and store them.

    Parameters
    ----------
    func : callable
        The function to be integrated.

    Returns
    -------
    out : class
        The array can be get by index or call directly.
    """
    def __init__(self, func, notes='', **kwargs):
        self.func = func
        self.notes = notes
        self.kwargs = kwargs
        self.array = {}

    def get_raw_array(self):
        return self.array
    
    def _cal(self, index):
        val = self.func(index, **self.kwargs)
        if isinstance(val, np.ndarray) and (val.size == 1):
            return val.item()
        return val

    def __getitem__(self, index):
        try: # if exist, return directly
            return self.array[index]
        except: # if not exist, calculate and store, then return
            self.array[index] = self._cal(index)
            return self.array[index]
        
    def __call__(self, index):
        """Allow the object to be called directly. Not recommanded to use this method."""
        return self.__getitem__(index)
        
    def __len__(self):
        return len(self.array)
        
    def __repr__(self):
        return f"Array_calculator({self.notes}, id: {id(self)})"


