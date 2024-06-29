import numpy as np
from ._collection import integral_method
from model.rect_lattice import eps_userdefine
import uuid, os, warnings, time
# filelock is not properly supported
# from filelock import FileLock


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
# The fucking thing is function cannot be pickled and at least a circular reference is created. I must use normal container class without function in class.
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
    def __init__(self, func, notes='',pathname_suffix='', **kwargs):
        self.func = func
        self.notes = notes
        self.kwargs = kwargs
        self.pathname = f'__temp_data__{pathname_suffix}'
        self.array = {'notes':notes,'kwargs':kwargs}
        self.unique_id = uuid.uuid4()
        self.array_path = f'./{self.pathname}/{self.unique_id}.npy'
        if not os.path.exists(f'./{self.pathname}/'):
            os.mkdir(f'./{self.pathname}/')
        np.save(self.array_path, self.array)

    def _cal(self, index):
        val = self.func(index, **self.kwargs)
        if isinstance(val, np.ndarray) and (val.size == 1):
            return val.item()
        return val
    
    def _update_array(self, index, value):
        try:
            self.array = np.load(self.array_path, allow_pickle=True).item()
            self.array[index] = value
            np.save(self.array_path, self.array)
        except:
            print('File may be occupied by other process. Try again')
            time.sleep(0)
            self._update_array(index, value)

    def __getitem__(self, index):
        try: # if exist, return directly
            self.array = np.load(self.array_path, allow_pickle=True).item()
            item = self.array[index]
            if item == 'placeholder':
                print(f'Calculating {index} is in progress...\nCheck again in 5 seconds.')
                time.sleep(5)
                warnings.warn(f'It will reduce the efficiency of the calculation. Please check the calculation progress.')
                return self.__getitem__(index)
            return item
        except: # if not exist, calculate and store, then return
            self._update_array(index, 'placeholder')
            item = self._cal(index)
            self._update_array(index, item)
            return item
        
    def __call__(self, index):
        """Allow the object to be called directly. Not recommanded to use this method."""
        return self.__getitem__(index)
        
    def __repr__(self):
        return f"Array_calculator({self.notes}, id: {id(self)})"

# not vectorized
class xi_calculator(Array_calculator):
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
        super().__init__(eps_func, notes=f'xi_calculator_{self.eps_type}_{self.method}', **kwargs)
    
    def _integrated_func(self, x, y, m:int, n:int):
        return self.eps_func(x, y)*np.exp(1j*(self.beta_0_x*m*x+self.beta_0_y*n*y))
    
    def _cal(self, index:tuple[int, int]):
        if self.eps_type == 'circle':
            self._xi = self._cal_circle(index) # maybe not need to assign?
        else:
            self._xi = self._cal_general(index)
        return self._xi
    
    def _cal_circle(self, index:tuple[int, int]):
        """Calculate the Fourier coefficients of the dielectric constant distribution for circle eps_func. Circle has discontinuity, so the integration should be separated into 3 zones."""
        print(f'xi: {index}')
        m, n = index
        if self.method == 'dblquad':
            integral_func = integral_method(3, method=self.method)()
            def boundary_yb(x):
                x = np.mod(x, self.eps_func.cell_size_x)
                if x < self.cell_size_x/2-self.eps_func.r or x > self.cell_size_x/2+self.eps_func.r:
                    return self.cell_size_y/2
                else:
                    return self.cell_size_y/2-np.sqrt(self.eps_func.r**2-(x-self.cell_size_x/2)**2)
            def boundary_yu(x):
                x = np.mod(x, self.eps_func.cell_size_x)
                if x < self.cell_size_x/2-self.eps_func.r or x > self.cell_size_x/2+self.eps_func.r:
                    return self.cell_size_y/2
                else:
                    return self.cell_size_y/2+np.sqrt(self.eps_func.r**2-(x-self.cell_size_x/2)**2)
            zone1 = integral_func(self._integrated_func, 0, self.cell_size_x, 0, boundary_yb, args=(m, n), **self.kwargs)
            zone2 = integral_func(self._integrated_func, 0, self.cell_size_x, boundary_yb, boundary_yu, args=(m, n), **self.kwargs)
            zone3 = integral_func(self._integrated_func, 0, self.cell_size_x, boundary_yu, self.cell_size_y, args=(m, n), **self.kwargs)
            zonels = [zone1, zone2, zone3]
            self._xi = np.sum([zone for zone in zonels])
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


class varsigma_matrix_calculator(Array_calculator):
    """
    Calculate the varsigma_matrix array and store them.

    Parameters
    ----------
    model : class Model, refer to model.Model
        The model class contains the parameters and functions.
    notes : str
        The notes of the array.
    pathname_suffix : str
        The suffix of the path name.
    **kwargs : keyword arguments, refer to example.py for details.
    """
    def __init__(self, model, notes='',pathname_suffix='', **kwargs):
        self.model = model
        self.notes = notes
        self.kwargs = kwargs
        self.pathname = f'__temp_data__{pathname_suffix}'
        self.array = {'notes':notes,'kwargs':kwargs}
        self.unique_id = uuid.uuid4()
        self.array_path = f'./{self.pathname}/{self.unique_id}.npy'
        if not os.path.exists(f'./{self.pathname}/'):
            os.mkdir(f'./{self.pathname}/')
        np.save(self.array_path, self.array)
        self._prepare_calculator()

    def _prepare_calculator(self):
        self._mu_func = self.model._mu_func
        self._nu_func = self.model._nu_func
        self.mu_calculator = Array_calculator(self.model._mu_func, notes='mu(index=(m,n,r,s))')
        self.nu_calculator = Array_calculator(self.model._nu_func, notes='nu(index=(m,n,r,s))')
        self.varsigma_matrix_calculator = Array_calculator(self._varsigma_matrix_func, notes='varsigma_matrix(index=(m,n))', **self.kwargs)
        
    def _cal(self, index):
        m, n = index
        return self.varsigma_matrix_calculator[m,n]
    
    def get_varsigma(self, order, direction:str):
        m, n, r, s = order
        r_s_order = np.where(self.r_s_order_ref == (r,s))
        if direction == 'x':
            return self.varsigma_matrix_calculator[m,n][0][r_s_order]
        elif direction == 'y':
            return self.varsigma_matrix_calculator[m,n][1][r_s_order]
        else:
            raise ValueError('direction must be x or y.')

    def __getattr__(self, name):
        if name == 'r_s_order_ref':
            self.r_s_order_ref = [(1,0),(-1,0),(0,1),(0,-1)]
            return self.r_s_order_ref
