import numpy as np
from scipy.optimize import Bounds, dual_annealing, minimize, direct
import warnings
from model.rect_lattice import eps_userdefine
vectorize_isinstance = np.vectorize(isinstance, excluded=['class_or_tuple'])
from coeff_func import xi_calculator, Array_calculator
from coeff_func import _dblquad_complex as dblquad_complex
from coeff_func import _quad_complex as quad_complex


class TMM():
    """
    Use TMM method to calculate the electrical field distribution of fundamental mode in a multilayer planar waveguide.

    Parameters
    ----------
    k0 : float, complex
        Wave number in free space.
    layer_thicknesses : iterable[float]
        Thickness of each layer.
    epsilons : iterable[float, complex]
        Dielectric constant of each layer.
    beta : float, complex
        Propagation constant of the fundamental mode.

    Returns
    -------
    out : class callable
        z : the position of the field, the z = 0 is the boundary of the first layer
        return : the E field normlized intensity. $$\int_{-\infty}^{\infty} |E_amp|^2 dz = 1$$
    """
    def __init__(self, layer_thicknesses, epsilons, beta, k0_init=None):
        self.find_modes_iter = 0
        self.k0_init = k0_init
        self.k0 = k0_init
        self.layer_thicknesses = np.array(layer_thicknesses, dtype=np.longdouble)
        self.epsilons = np.array(epsilons, dtype=np.longcomplex)
        self.beta = np.array(beta, dtype=np.longcomplex)
        self.z_boundary = np.insert(np.cumsum(self.layer_thicknesses), 0, 0) # the z position of the boundary of each layer
        self._z_boundary_without_lb = self.z_boundary[1:]
        self._len_z_boundary = len(self.z_boundary)

    def _construct_matrix(self, k0):
        self.gamma_s = np.sqrt(np.square(self.beta)-np.square(k0)*self.epsilons)
        prime = self.gamma_s[:-1]*self.layer_thicknesses[:-1]
        coeff_s = self.gamma_s[:-1]/self.gamma_s[1:]
        T_mat_s_00 = (1+coeff_s)*np.exp(prime)/2
        T_mat_s_01 = (1-coeff_s)*np.exp(-prime)/2
        T_mat_s_10 = (1-coeff_s)*np.exp(prime)/2
        T_mat_s_11 = (1+coeff_s)*np.exp(-prime)/2
        self.T_mat_s = np.zeros((len(self.layer_thicknesses)-1,2,2), dtype=np.longcomplex) # Must define the dtype, otherwise the default dtype is float64, which will cause error in assign complex number to float64.
        self.T_mat_s[:,0,0] = T_mat_s_00
        self.T_mat_s[:,0,1] = T_mat_s_01
        self.T_mat_s[:,1,0] = T_mat_s_10
        self.T_mat_s[:,1,1] = T_mat_s_11
        self.T_total = self._matrix_multiply(self.T_mat_s)
        self.t_11 = self.T_total[0,0]

    def _matrix_multiply(self, mat_s):
        """
        multiply the matrix in the list by reverse order
        mat_s: the matrix list
        return: the matrix multiply result
        """
        mat = np.eye(2, dtype=np.longcomplex)
        V_i = np.array([[1],[0]])
        self.V_s = [V_i,]
        for i in range(len(mat_s)):
            mat = np.dot(mat_s[i], mat)
            V_i = np.dot(mat_s[i], V_i)
            self.V_s.append(V_i)
        return mat
    
    def _find_layer(self, z):
        """
        Parameters
        ----------
        z : float
            The position of the field, the z = 0 is the boundary of the first layer

        Returns
        -------
        out : int
            The index of the layer.
        """
        __find_layer = np.vectorize(self.__find_layer)
        return __find_layer(z)
    
    def __find_layer(self, z):
        i = 0
        while z >= self._z_boundary_without_lb[i] and i <= self._len_z_boundary-3:
            i += 1
        return i  
    
    def e_amplitude(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field amplitude
        """
        def _e_amplitude(z):
            i = self._find_layer(z)
            V = self.V_s[i].flatten()
            z_b = self.z_boundary[i]
            e_amp = V[0]*np.exp(self.gamma_s[i]*(z-z_b))+V[1]*np.exp(-self.gamma_s[i]*(z-z_b))
            eps = self.epsilons[i]
            return e_amp, eps
        _e_amplitude = np.vectorize(_e_amplitude)
        return _e_amplitude(z)
    
    def e_normlized_intensity(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field normlized intensity. $$\int_{-\infty}^{\infty} |E_amp|^2 dz = 1$$
        """
        e_amp, eps = self.e_amplitude(z)
        e_intensity = np.square(np.abs(e_amp))*self._normlized_constant
        return e_intensity, eps
    
    def e_normlized_amplitude(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field normlized amplitude. $$\int_{-\infty}^{\infty} |E_amp|^2 dz = 1$$
        """
        e_amp, eps = self.e_amplitude(z)
        e_norm_amp = e_amp*np.sqrt(self._normlized_constant)
        return e_norm_amp, eps
    
    def find_modes(self):
        """
        return: the k0 of the mode
        """
        def t_11_func_k_log(k0):
            self._construct_matrix(k0)
            log_t11 = np.log10(np.abs(self.t_11))
            return log_t11
        if self.k0_init is None:
            k0_min = np.array(np.real(self.beta/np.sqrt(np.real(self.epsilons).max())), dtype=np.longdouble) # k0 must be smaller than wave in highest epsilon medium
            k0_max = np.array(np.real(self.beta/np.sqrt(np.real(self.epsilons)[0])), dtype=np.longdouble) # k0 must be bigger than wave in cladding medium
            k_sol_coarse = direct(t_11_func_k_log, Bounds(k0_min,k0_max))
            k0_init = k_sol_coarse.x
        else:
            k0_init = self.k0_init
        k_sol = minimize(t_11_func_k_log, x0=k0_init, method='Nelder-Mead')
        while k_sol.fun > -14.0 and self.find_modes_iter < 5:
            self.find_modes_iter += 1
            if k_sol.fun < -10.0:
                self.k0_init = k_sol.x
            warnings.warn(f't11 is larger than 1e-14 after {k_sol.nit} iter, t11 = {10**k_sol.fun}. Retry {self.find_modes_iter}.', RuntimeWarning)
            self.find_modes()
        self.k0 = k_sol.x
        self._construct_matrix(self.k0)
        self._cal_normlized_constant()

    def _cal_normlized_constant(self):
        from scipy.integrate import quad
        e_amp_min_r = minimize(lambda z: np.abs(np.real(self.e_amplitude(z)[0])), x0=self.z_boundary[-1], method='Nelder-Mead')
        e_amp_min_l = minimize(lambda z: np.abs(np.real(self.e_amplitude(z)[0])), x0=self.z_boundary[0], method='Nelder-Mead')
        e_amp_min_boundary_distance = min(np.abs(e_amp_min_r.x-self.z_boundary[-1]), np.abs(e_amp_min_l.x-self.z_boundary[0]))
        self._normlized_bd = e_amp_min_boundary_distance
        e_amp_integral = quad(lambda z: np.square(np.real(self.e_amplitude(z)[0])), self.z_boundary[0]-self._normlized_bd, self.z_boundary[-1]+self._normlized_bd)
        self._normlized_constant = 1/e_amp_integral[0]

    def __getattr__(self, name):
        if name == '_normlized_constant':
            try:
                return self._normlized_constant
            except:
                warnings.warn('The normlized constant is not calculated yet, because find_modes is not called.')
                print('Force calculate normlized constant now...')
                self._cal_normlized_constant()

    def __call__(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field normlized intensity. $$\int_{-\infty}^{\infty} |E_amp|^2 dz = 1$$
        """
        return self.e_normlized_intensity(z)
    

class model_parameters():
    """
    Parameters
    ----------
    layer : tuple[list[float],list[complex|eps_userdefine]]
        The layer information of the model.
        The first element is the thickness of each layer.
        The second element is the dielectric constant of each layer. If the dielectric constant is eps_userdefine, the cell size of the eps_userdefine must be the same.
    **kwargs : dict
        The other parameters of the model.

    Returns
    -------
    out : class
        The model parameters.
    """
    def __init__(self, layer:tuple[list[float],list[complex|eps_userdefine]], **kwargs):
        self.layer_thicknesses = np.array(layer[0])
        self.epsilons = np.array(layer[1])
        self.kwargs = kwargs
        self._check_para()
        self.beta = 2*np.pi/np.sqrt(self.cellsize_x*self.cellsize_y)
        self._gen_avg_eps()

    def _check_para(self):
        cellsize_x = []
        cellsize_y = []
        for _ in self.epsilons:
            if isinstance(_, eps_userdefine):
                cellsize_x.append(_.cell_size_x)
                cellsize_y.append(_.cell_size_y)
        if len(set(cellsize_x)) == 0: raise ValueError("At least one eps_userdefine is needed in epsilons.")
        assert len(set(cellsize_x)) == 1, 'The cellsize_x of eps_userdefine must be the same.'
        assert len(set(cellsize_y)) == 1, 'The cellsize_y of eps_userdefine must be the same.'
        self.cellsize_x = cellsize_x[0]
        self.cellsize_y = cellsize_y[0]

    def _gen_avg_eps(self):
        avg_eps = []
        for _ in self.epsilons:
            if isinstance(_, eps_userdefine):
                avg_eps.append(_.avg_eps)
            else:
                avg_eps.append(_)
        self.avg_epsilons = np.array(avg_eps)

class Model():
    """
    Parameters
    ----------
    paras : class model_parameters
        The model parameters.

    Returns
    -------
    out : class callable
        x, y, z : the position. When called, return epsilon. If not given x, y, average epsilon is return.
    """
    def __init__(self, paras:model_parameters):
        self.paras = paras
        self.tmm = TMM(paras.layer_thicknesses, paras.avg_epsilons, paras.beta, paras.kwargs.get('k0_init', None))
        self.tmm.find_modes()
        self.k0 = self.tmm.k0
        self.beta0 = self.tmm.beta
        self.prepare_calculator()

    def e_profile(self, z):
        return self.tmm(z)
    
    def eps_profile(self, x=None, y=None, z=None):
        if z is None:
            raise RuntimeError('z could not be None.')
        if (x is None) or (y is None):
            return self.__eps_profile_z(z=z)
        x = np.array(x, dtype=np.longdouble)
        y = np.array(y, dtype=np.longdouble)
        z = np.array(z, dtype=np.longdouble)
        if x.shape != y.shape:
            raise ValueError('The shape of x, y must be the same.')
        return self.__eps_profile(x, y, z)

    def __eps_profile(self, x, y, z):
        num_layer = self.tmm._find_layer(z)
        eps = self.paras.epsilons[num_layer]
        is_eps_userdefine = vectorize_isinstance(eps, eps_userdefine)
        eps_array = np.empty_like(z, dtype=object)
        for i in range(len(z)):
            if is_eps_userdefine[i]:
                eps_array[i] = eps[i](x, y)
            else:
                eps_array[i] = np.ones_like(x)*eps[i]
        return eps_array
    
    def __eps_profile_z(self, z):
        num_layer = self.tmm._find_layer(z)
        return self.paras.avg_epsilons[num_layer]
    
    def is_in_phc(self, z):
        def _is_in_phc(z):
            num_layer = self.tmm._find_layer(z)
            if isinstance(self.paras.epsilons[num_layer], eps_userdefine):
                return True
            else:
                return False
        _is_in_phc = np.vectorize(_is_in_phc)
        return _is_in_phc(z)  

    def __call__(self, x=None, y=None, z=None):
        return self.eps_profile(x, y, z)
    
    def prepare_calculator(self):
        self.xi_calculator_collect = []
        for _ in self.paras.epsilons:
            if isinstance(_, eps_userdefine):
                self.xi_calculator_collect.append(xi_calculator(_))
            else:
                self.xi_calculator_collect.append(None)
        self.mu_calculator = Array_calculator(self.mu_func, notes='mu(index=(m,n,r,s))')
        self.nu_calculator = Array_calculator(self.nu_func, notes='nu(index=(m,n,r,s))')
        
    
    def Green_func_fundamental(self, z, z_prime):
        # Approximatly Green function
        return -1j/(2*self.beta_z_func_fundamental(z))*np.exp(-1j*self.beta_z_func_fundamental(z)*np.abs(z-z_prime))
    
    def beta_z_func_fundamental(self, z):
        return self.k0*self.__eps_profile_z(z)
    
    def Green_func_higher_order(self, z, z_prime, order):
        # Approximatly Green function of higher order
        return -1j/(2*self.beta_z_func_higher_order(z, order))*np.exp(-1j*self.beta_z_func_higher_order(z, order)*np.abs(z-z_prime))
    
    def beta_z_func_higher_order(self, z, order):
        # TODO: Check the formula. The formulas may be in the wrong order.
        m, n = order
        return np.sqrt( (np.square(m)+np.square(n))*np.square(self.beta0) - np.square(self.beta_z_func_fundamental(z)) )
    
    def xi_z_func(self, z, order):
        m, n = order
        i = self.tmm._find_layer(z)
        if self.xi_calculator_collect[i] is not None:
            return self.xi_calculator_collect[i][order]
        else:
            return 0+0j

    def mu_func(self, index):
        m, n, r, s = index
        def integrated_func(z, z_prime):
            return self.xi_z_func(z_prime,(m-r,n-s))*self.Green_func_higher_order(z,z_prime,(m,n))*self.tmm.e_normlized_amplitude(z_prime)[0]*np.conj(self.tmm.e_normlized_amplitude(z)[0])
        return 1/np.square(self.k0)*dblquad_complex(integrated_func, self.tmm.z_boundary[0], self.tmm.z_boundary[-1])[0]
    
    def nu_func(self, index):
        m, n, r, s = index
        def integrated_func(z):
            return 1/self.__eps_profile_z(z)*self.xi_z_func(z,(m-r,n-s))*self.tmm.e_normlized_intensity(z)[0]
        return -quad_complex(integrated_func, self.tmm.z_boundary[0], self.tmm.z_boundary[-1])[0]
    