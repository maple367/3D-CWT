import numpy as np
from scipy.integrate import dblquad, trapezoid, simpson, romb, qmc_quad
from model.rect_lattice import eps_userdefine


class xi_func_DFT():
    """
    Calculate the Fourier coefficients of the dielectric constant distribution.
    Use the Discrete Fourier Transform (DFT) to calculate the Fourier coefficients.
    Parameters
    ----------
    eps_func : class eps_poly
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


class xi_func():
    """
    Calculate the Fourier coefficients of the dielectric constant distribution.
    Parameters
    ----------
    eps_func : class eps_poly
        The class contains dielectric constant distribution in 2D plane and cell size.
    method : str, {'dbltrapezoid', 'dblsimpson', 'dblquad', 'dblromb', 'dblqmc_quad'}
        The size of the cell in the y direction.
    **kwargs : keyword arguments, refer to example.py for details.

    Returns
    -------
    out : class
        The Fourier coefficients.
    """
    def __init__(self, eps_func:eps_userdefine, method='dblromb', **kwargs):
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
    
    def _dblquad_complex(self, func, a, b, gfun, hfun, args=(), **kwargs):
        """Double quad integration. Suitable for real number space integral routine.
        The function should return a complex number and abserr."""
        def real_func(y, x, *args):
            return np.real(func(x, y, *args))
        def imag_func(y, x, *args):
            return np.imag(func(x, y, *args))
        real_integral = dblquad(real_func, a, b, gfun, hfun, args=args, **kwargs)
        imag_integral = dblquad(imag_func, a, b, gfun, hfun, args=args, **kwargs)
        return real_integral[0] + 1j*imag_integral[0], real_integral[1] + 1j*imag_integral[1]
    
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
        m, n = index
        match self.method:
            case 'dbltrapezoid':
                print('dbltrapezoid not supported for circle eps_func now. Use userdefine instead.')
                self.eps_type = 'userdefine'
                return self._cal(index)
            case 'dblsimpson':
                print('dblsimpson not supported for circle eps_func now. Use userdefine instead.')
                self.eps_type = 'userdefine'
                return self._cal(index)
            case 'dblquad':
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
                zone1 = self._dblquad_complex(self._integrated_func, 0, self.cell_size_x, 0, boundary_yb, args=(m, n), **self.kwargs)
                zone2 = self._dblquad_complex(self._integrated_func, 0, self.cell_size_x, boundary_yb, boundary_yu, args=(m, n), **self.kwargs)
                zone3 = self._dblquad_complex(self._integrated_func, 0, self.cell_size_x, boundary_yu, self.cell_size_y, args=(m, n), **self.kwargs)
                zonels = [zone1, zone2, zone3]
                self._xi = sum([zone[0] for zone in zonels])
                self._abserr = sum([np.abs(np.real(zone[1]))+1j*np.abs(np.imag(zone[1])) for zone in zonels])
            case 'dblromb':
                print('dblromb not supported for circle eps_func now. Use userdefine instead.')
                self.eps_type = 'userdefine'
                return self._cal(index)
            case 'dblqmc_quad':
                print('dblqmc_quad not supported for circle eps_func now. Use userdefine instead.')
                self.eps_type = 'userdefine'
                return self._cal(index)
            case _:
                raise ValueError('Method not supported.')
        self._xi = self._xi/(self.cell_size_x*self.cell_size_y)
        return self._xi
    
    def _cal_general(self, index:tuple[int, int]):
        m, n = index
        match self.method:
            case 'dbltrapezoid':
                x_mesh = np.linspace(0, self.cell_size_x, num=self.kwargs['resolution'])
                y_mesh = np.linspace(0, self.cell_size_y, num=self.kwargs['resolution'])
                XX, YY = np.meshgrid(x_mesh, y_mesh, indexing='ij')
                self._xi = self._dbltrapezoid(self._integrated_func, XX, YY, args=(m, n))
            case 'dblsimpson':
                x_mesh = np.linspace(0, self.cell_size_x, num=self.kwargs['resolution'])
                y_mesh = np.linspace(0, self.cell_size_y, num=self.kwargs['resolution'])
                XX, YY = np.meshgrid(x_mesh, y_mesh, indexing='ij')
                self._xi = self._dblsimpson(self._integrated_func, XX, YY, args=(m, n))
            case 'dblquad':
                self._xi, self._abserr = self._dblquad_complex(self._integrated_func, 0, self.cell_size_x, 0, self.cell_size_y, args=(m, n), **self.kwargs)
            case 'dblromb':
                x_mesh = np.linspace(0, self.cell_size_x, num=self.kwargs['resolution'], retstep=True)
                y_mesh = np.linspace(0, self.cell_size_y, num=self.kwargs['resolution'], retstep=True)
                XX, YY = np.meshgrid(x_mesh[0], y_mesh[0], indexing='ij')
                self._xi = self._dblromb(self._integrated_func, XX, YY, x_mesh[1], y_mesh[1], args=(m, n))
            case 'dblqmc_quad':
                self._xi, self._abserr = self._dblqmc_quad(self._integrated_func, 0, self.cell_size_x, 0, self.cell_size_y, args=(m, n), **self.kwargs)
            case _:
                raise ValueError('Method not supported.')
        self._xi = self._xi/(self.cell_size_x*self.cell_size_y)
        return self._xi

    def get_raw_array(self):
        return self.xi_array


class TMM_cal():
    # """
    # k0: wave number in vacuum
    # t_s: thickness of each layer, list or array
    # eps_s: epsilon of each layer, list or array
    # The layer structure is looked like:

    # --|eps1,t0|eps2,t1|eps3,t2|--
    # where eps0 and eps4 are the epsilon of the two boundarys, eps1, eps2, eps3 are the epsilon of the three layers, t0, t1, t2 are the thickness of the three layers.
    # return: the value needed for constructing matrix of TMM
    # """
    """
    Use TMM method to calculate the electrical field distribution of fundamental mode.
    Parameters
    ----------
    k0 : float, complex
        Wave number in free space.


    Returns
    -------
    out : class
        The Fourier coefficients.
    """
    def __init__(self, k0, t_s, eps_s, beta):
        self.k0 = k0
        self.t_s = np.array(t_s, dtype=np.float64)
        self.eps_s = np.array(eps_s, dtype=np.complex128)
        self.beta = np.array(beta,dtype=np.complex128)
        self.__construct_matrix()
        T_total= self.__matrix_multiply(self.T_mat_s)
        self.t_11 = T_total[0,0]

    def __construct_matrix(self):
        gamma_s = np.sqrt(np.square(self.beta)-np.square(self.k0)*self.eps_s)
        self.gamma_s = gamma_s
        prime = gamma_s[:-1]*self.t_s[:-1]
        coeff_s = gamma_s[:-1]/gamma_s[1:]
        T_mat_s_00 = (1+coeff_s)*np.exp(prime)/2
        T_mat_s_01 = (1-coeff_s)*np.exp(-prime)/2
        T_mat_s_10 = (1-coeff_s)*np.exp(prime)/2
        T_mat_s_11 = (1+coeff_s)*np.exp(-prime)/2
        self.T_mat_s = np.zeros((len(self.t_s)-1,2,2), dtype=np.complex128)
        self.T_mat_s[:,0,0] = T_mat_s_00
        self.T_mat_s[:,0,1] = T_mat_s_01
        self.T_mat_s[:,1,0] = T_mat_s_10
        self.T_mat_s[:,1,1] = T_mat_s_11

    def __matrix_multiply(self, mat_s):
        """
        multiply the matrix in the list by reverse order
        mat_s: the matrix list
        return: the matrix multiply result
        """
        mat = np.eye(2, dtype=np.complex128)
        V_i = np.array([[1],[0]])
        self.V_s = [V_i,]
        for i in range(len(mat_s)):
            mat = np.dot(mat_s[i], mat)
            V_i = np.dot(mat_s[i], V_i)
            self.V_s.append(V_i)
        return mat
    
    def __find_layer(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the index of the layer
        """
        z_s = np.cumsum(self.t_s)
        for i in range(len(z_s)):
            if z < z_s[i]:
                break
        return i
    
    def E_field(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field amplitude
        """
        def E_field_s(z):
            i = self.__find_layer(z)
            z_s = np.insert(np.cumsum(self.t_s), 0, 0)
            V = self.V_s[i].flatten()
            z_b = z_s[i]
            E_amp = V[0]*np.exp(self.gamma_s[i]*(z-z_b))+V[1]*np.exp(-self.gamma_s[i]*(z-z_b))
            eps = self.eps_s[i]
            return E_amp, eps
        return np.vectorize(E_field_s)(z)

def find_k0(beta_0, t_ls, eps_ls, run_time=0):
    # optimze function
    from scipy.optimize import Bounds, dual_annealing, minimize
    def t11_func_k(k0, beta_0):
        t11 = TMM_cal(k0, t_ls, eps_ls, beta_0).t_11
        return np.log10(np.abs(t11))
    k_min = beta_0/np.sqrt(np.real(eps_ls).max()) # k0 must be smaller than wave in highest RI medium
    k_max = beta_0/np.sqrt(np.real(eps_ls)[0]) # k0 must be bigger than wave in cladding medium
    k_sol = dual_annealing(t11_func_k, Bounds(k_min,k_max), args=(beta_0,)) # Using dual_annealing to find a coarse result.
    k_sol = minimize(t11_func_k, k_sol.x, args=(beta_0,), method='Nelder-Mead') # Using minimize to find a more exactly result.
    if k_sol.fun > -10.0:
        RuntimeWarning(f'Warning: the t11 is not smaller than 1e-10, t11 = {np.power(10, k_sol.fun)}. Retry')
        run_time += 1
        if run_time > 3:
            raise RuntimeError(f'Error: already retried 3 times.')
        find_k0(beta_0, t_ls, eps_ls, run_time)
    return k_sol.x

if __name__ == '__main__':
    # ignore the RuntimeError
    import warnings
    warnings.filterwarnings('ignore')
    FF_lst = np.linspace(0.05,0.40,21)
    k0_lst = []
    for FF in FF_lst:
        beta_0 = 2*np.pi/0.295
        t_list = [1.5,0.0885,0.1180,0.0590,1.5]
        eps_list = [11.0224,12.8603,FF+(1-FF)*12.7449,12.7449,11.0224]
        k0 = find_k0(beta_0, t_list, eps_list)
        k0_lst.append(k0)
        # import matplotlib.pyplot as plt
        # k0_lst = np.array(k0_lst)
        # lambda0_lst = 2*np.pi/k0_lst
        # plt.plot(FF_lst, lambda0_lst)
        # plt.show()

        import matplotlib.pyplot as plt
        from scipy.optimize import Bounds, dual_annealing, minimize
        def t11_func_k(lambda0, beta):
            k0 = 2*np.pi/lambda0
            t11 = TMM_cal(k0, t_list, eps_list, beta).t_11
            return np.log10(np.abs(t11))
        t11_func_k = np.vectorize(t11_func_k)

        def t11_func_beta(beta, lambda0):
            k0 = 2*np.pi/lambda0
            t11 = TMM_cal(k0, t_list, eps_list, beta).t_11
            return np.log10(np.abs(t11))


        lambda0_min = 0.295*np.sqrt(np.real(eps_list)[0])
        lambda0_max = 0.295*np.sqrt(np.real(eps_list).max())
        lambda0 = np.linspace(lambda0_min, lambda0_max, 5000)
        t_l = t11_func_k(lambda0, beta_0)
        # plt.plot(lambda0, t_l)
        # plt.show()
        k_sol = dual_annealing(t11_func_k, Bounds(lambda0_min,lambda0_max), args=(beta_0,))
        k_sol = minimize(t11_func_k, k_sol.x, args=(beta_0,), method='Nelder-Mead')
        # beta_sol = dual_annealing(t11_func_beta, Bounds(2*np.pi/0.32,2*np.pi/0.28), args=(k_sol.x,))
        # beta_sol = minimize(t11_func_beta, beta_0, args=(k_sol.x,), method='Nelder-Mead')

        z_mesh = np.linspace(-1, 1.5 + 0.0885 + 0.1180 + 0.0590 + 1.5 +1, 5000)
        E_field_s, eps_s = TMM_cal(2*np.pi/k_sol.x, t_list, eps_list, beta_0).E_field(z_mesh)
        E_field_s = np.array(E_field_s)
        plt.plot(z_mesh, (np.real(E_field_s)/np.max(np.real(E_field_s)))**2, 'b-')
        ax1 = plt.twinx()
        plt.ylabel('eps')
        plt.title(f'k_0 = {k0}, beta = {beta_0}, t11 = {np.power(10, k_sol.fun)}')
        plt.plot(z_mesh, eps_s, 'r--')
        plt.show()