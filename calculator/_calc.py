import numpy as np
from ._collection import integral_method, Array_calculator
from model.rect_lattice import eps_userdefine
import numba
import time



class Array_calculator_central_symmetry(Array_calculator):
    def _update_array(self, index, value):
        while True:
            try:
                with self.lock:
                    self.array = np.load(self.array_path, allow_pickle=True).item()
                    self.array[index] = value
                    
                    if value == 'placeholder':
                        self.array[tuple([-num for num in index])] = 'placeholder'
                    else:
                        self.array[tuple([-num for num in index])] = np.conj(value)
                    np.save(self.array_path, self.array)
                    return
            except Exception as e:
                print(f'Fail in updating {self.notes} {index} with value {value}. Try again.')
                time.sleep(0.1)


@numba.njit(cache=True)
def __xi__integrated_func__(val, x, y, m, n, beta_0_x, beta_0_y):
    return val*np.exp(1j*(beta_0_x*m*x+beta_0_y*n*y))

# not vectorized
class xi_calculator(Array_calculator_central_symmetry):
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
    def __init__(self, eps_func:eps_userdefine, notes, pathname_suffix, method='dblquad', **kwargs):
        self.eps_func = eps_func
        self.eps_type = self.eps_func.eps_type
        self.method = method
        self.kwargs = kwargs
        self.cell_size_x = eps_func.cell_size_x
        self.cell_size_y = eps_func.cell_size_y
        self.beta_0_x = 2*np.pi/self.cell_size_x
        self.beta_0_y = 2*np.pi/self.cell_size_y
        super().__init__(eps_func, notes, pathname_suffix, **kwargs)
        self._check_method()

    def _check_method(self):
        if self.method not in ['dbltrapezoid', 'dblsimpson', 'dblquad', 'dblromb', 'dblqmc_quad']:
            raise ValueError('Method not supported.')

    def _integrated_func(self, x, y, m:int, n:int):
        val = self.eps_func.eps(x, y)
        return __xi__integrated_func__(val, x, y, m, n, self.beta_0_x, self.beta_0_y)
    
    def _cal(self, index:tuple[int, int]):
        print(f'\rxi: {index}  ', end='', flush=True)
        if self.eps_type == 'CC':
            self._xi = self._cal_circle(index) # TODO: maybe not need to assign?
        elif self.eps_type == 'RIT':
            self._xi = self._cal_ritriangle(index)
        else:
            self._xi = self._cal_general(index)
        return self._xi

    def _cal_circle(self, index:tuple[int, int]):
        """Calculate the Fourier coefficients of the dielectric constant distribution for circle eps_func. Circle has discontinuity, so the integration should be separated into 3 zones."""
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
        self._xi = self._xi/(self.cell_size_x*self.cell_size_y)
        return self._xi

    def _cal_ritriangle(self, index:tuple[int, int]):
        """Calculate the Fourier coefficients of the dielectric constant distribution for right-angled isosceles triangle eps_func. right-angled isosceles triangle has discontinuity, so the integration should be separated into 3 zones."""
        m, n = index
        if self.method == 'dblquad':
            integral_func = integral_method(3, method=self.method)()
            def boundary_yb(x):
                x = np.mod(x, self.eps_func.cell_size_x)
                if x < self.cell_size_x/2-self.eps_func.s/2 or x > self.cell_size_x/2+self.eps_func.s/2:
                    return -self.cell_size_y/self.cell_size_x*x+self.cell_size_y
                else:
                    return self.cell_size_y/2-self.eps_func.s/2
            def boundary_yu(x):
                x = np.mod(x, self.eps_func.cell_size_x)
                return -self.cell_size_y/self.cell_size_x*x+self.cell_size_y
            zone1 = integral_func(self._integrated_func, 0, self.cell_size_x, 0, boundary_yb, args=(m, n), **self.kwargs)
            zone2 = integral_func(self._integrated_func, 0, self.cell_size_x, boundary_yb, boundary_yu, args=(m, n), **self.kwargs)
            zone3 = integral_func(self._integrated_func, 0, self.cell_size_x, boundary_yu, self.cell_size_y, args=(m, n), **self.kwargs)
            zonels = [zone1, zone2, zone3]
            self._xi = np.sum([zone for zone in zonels])
        elif self.method in ['dbltrapezoid', 'dblsimpson', 'dblromb', 'dblqmc_quad']:
            print(f'{self.method} not supported for ritriangle eps_func now. Use userdefine instead.')
            self.eps_type = 'userdefine'
            return self._cal(index)
        self._xi = self._xi/(self.cell_size_x*self.cell_size_y)
        return self._xi
    
    def _cal_general(self, index:tuple[int, int]):
        integral_func = integral_method(3, method=self.method)()
        m, n = index
        self._xi = integral_func(self._integrated_func, 0, self.cell_size_x, 0, self.cell_size_y, args=(m, n), **self.kwargs)
        self._xi = self._xi/(self.cell_size_x*self.cell_size_y)
        return self._xi

class xi_calculator_DFT(xi_calculator):
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
    def __init__(self, eps_func:eps_userdefine, notes='', pathname_suffix='', resolution:int=23, subresolution=10, **kwargs):
        self.eps_func = eps_func
        self.resolution = resolution
        self.subresolution = subresolution
        self._build_array()
        super().__init__(eps_func, notes, pathname_suffix, **kwargs)

    def _build_array(self):
        self.eps_array = np.empty((self.resolution, self.resolution), dtype=complex)
        grid_size_x = self.eps_func.cell_size_x/self.resolution
        grid_size_y = self.eps_func.cell_size_y/self.resolution
        x_c = np.arange(0, self.eps_func.cell_size_x, grid_size_x) + grid_size_x/2
        y_c = np.arange(0, self.eps_func.cell_size_y, grid_size_y) + grid_size_y/2
        XX, YY = np.meshgrid(x_c, y_c)
        for i in range(self.resolution):
            for j in range(self.resolution):
                xx_c = XX[i,j] + np.arange(-self.subresolution/2+1/2, self.subresolution/2+1/2)*grid_size_x/self.subresolution
                yy_c = YY[i,j] + np.arange(-self.subresolution/2+1/2, self.subresolution/2+1/2)*grid_size_y/self.subresolution
                self.eps_array[i,j] = np.average(self.eps_func.eps(xx_c, yy_c))
        self.xi_array = np.fft.fft2(self.eps_array)/(self.resolution**2)

    def _cal(self, index:tuple[int, int]):
        print(f'\rxi: {index}  ', end='', flush=True)
        try:
            self.array[index] = self.xi_array[index]
        except:
            if self.resolution < 2*np.max(np.abs(index))+1:
                self.resolution = 2*np.max(np.abs(index))+1
                self._build_array()
            self._xi = self.xi_array[index]
        return self._xi
    
    def set_resolution(self, resolution:int):
        self.resolution = resolution
        self._build_array()


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
        self.pathname_suffix = pathname_suffix
        super().__init__(model, notes=notes, pathname_suffix=pathname_suffix, **kwargs)
        self._prepare_calculator()
        self.r_s_order_ref = [(1,0),(-1,0),(0,1),(0,-1)]

    def _prepare_calculator(self):
        self._mu_func = self.model._mu_func
        self._nu_func = self.model._nu_func
        self.mu_calculator = Array_calculator_central_symmetry(self.model._mu_func, notes='mu(index=(m,n,r,s))', pathname_suffix=self.pathname_suffix, lock=self.lock)
        self.nu_calculator = Array_calculator_central_symmetry(self.model._nu_func, notes='nu(index=(m,n,r,s))', pathname_suffix=self.pathname_suffix, lock=self.lock)
        
    def _cal(self, index):
        print(f'\rvarsigma_matrix: {index}  ', end='', flush=True)
        m, n = index
        mat1 = np.array([[n, m],
                         [-m, n]])
        mat2 = np.array([[-m*self.mu_calculator[m,n,1,0], -m*self.mu_calculator[m,n,-1,0], n*self.mu_calculator[m,n,0,1], n*self.mu_calculator[m,n,0,-1]],
                         [n*self.nu_calculator[m,n,1,0], n*self.nu_calculator[m,n,-1,0], m*self.nu_calculator[m,n,0,1], m*self.nu_calculator[m,n,0,-1]]])
        return 1/(np.square(m)+np.square(n))*np.dot(mat1, mat2)
    
    def get_varsigma(self, order, direction:str):
        m, n, r, s = order
        r_s_order = self.r_s_order_ref.index((r,s))
        varsigma_matrix = self[m,n]
        if direction == 'x':
            return varsigma_matrix[0][r_s_order]
        elif direction == 'y':
            return varsigma_matrix[1][r_s_order]
        else:
            raise ValueError('direction must be x or y.')
        
