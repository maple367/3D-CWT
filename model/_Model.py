import numpy as np
from scipy.optimize import Bounds, dual_annealing, minimize, direct
from scipy.integrate import quad
import numba
import warnings
from model.rect_lattice import eps_userdefine
vectorize_isinstance = np.vectorize(isinstance, excluded=['class_or_tuple'])
import multiprocessing as mp
import dill
dill.extend(use_dill=True)


@numba.njit(cache=True)
def __find_layer__(z, _z_boundary_without_lb, _len_z_boundary_2):
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
    res = np.searchsorted(_z_boundary_without_lb, z, side='right')
    return np.where(z >= _z_boundary_without_lb[-1], _len_z_boundary_2, res)


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
        return __find_layer__(z, self._z_boundary_without_lb, self._len_z_boundary-2)
    
    def e_amplitude(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field amplitude
        """
        i = self._find_layer(z)
        V = self.V_s_flatten[i]
        z_b = self.z_boundary[i]
        V_0 = np.take(V,0,axis=-1)
        V_1 = np.take(V,1,axis=-1)
        e_amp = V_0*np.exp(self.gamma_s[i]*(z-z_b))+V_1*np.exp(-self.gamma_s[i]*(z-z_b))
        eps = self.epsilons[i]
        return e_amp, eps
    
    def e_normlized_intensity(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field normlized intensity. $$\int_{-\infty}^{\infty} |E_amp|^2 dz = 1$$
        """
        e_amp, _ = self.e_amplitude(z)
        e_intensity = np.square(np.abs(e_amp))*self._normlized_constant
        return e_intensity
    
    def e_normlized_amplitude(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field normlized amplitude. $$\int_{-\infty}^{\infty} |E_amp|^2 dz = 1$$
        """
        e_amp, _ = self.e_amplitude(z)
        e_norm_amp = e_amp*np.sqrt(self._normlized_constant)
        return e_norm_amp
    
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
        while (k_sol.fun >= -10.0) and self.find_modes_iter < 3:
            self.find_modes_iter += 1
            if k_sol.fun < -10.0:
                self.k0_init = k_sol.x
            warnings.warn(f't11 is not converge to machine precision after {k_sol.nit} iter, t11 = {10**k_sol.fun}. Retry {self.find_modes_iter}.', RuntimeWarning)
            self.find_modes()
        self.k0 = k_sol.x
        self._construct_matrix(self.k0)
        self._cal_normlized_constant()
        self.V_s_flatten = np.array([_.flatten() for _ in self.V_s])

    def _cal_normlized_constant(self):
        def e_amp_real_func(z):
            return np.abs(np.real(self.e_amplitude(z)[0]))
        def e_amp_square_func(z):
            return np.square(np.abs(self.e_amplitude(z)[0]))
        e_amp_min_r = minimize(e_amp_real_func, x0=self.z_boundary[-1], method='Nelder-Mead')
        e_amp_min_l = minimize(e_amp_real_func, x0=self.z_boundary[0], method='Nelder-Mead')
        e_amp_min_boundary_distance = min(np.abs(e_amp_min_r.x-self.z_boundary[-1]), np.abs(e_amp_min_l.x-self.z_boundary[0]))
        self._normlized_bd = e_amp_min_boundary_distance
        e_amp_integral = quad(e_amp_square_func, self.z_boundary[0]-self._normlized_bd, self.z_boundary[-1]+self._normlized_bd)
        self._normlized_constant = 1/e_amp_integral[0]

    def __getattr__(self, name):
        if name == '_normlized_constant':
            try:
                return self._normlized_constant
            except:
                warnings.warn('The normlized constant is not calculated yet, because find_modes is not called.')
                print('Force calculate normlized constant now...')
                self._cal_normlized_constant()
        elif name == 'V_s_flatten':
            self.V_s_flatten = np.array([_.flatten() for _ in self.V_s])
            return self.V_s_flatten

    def __call__(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field normlized intensity. $$\int_{-\infty}^{\infty} |E_amp|^2 dz = 1$$
        """
        return self.e_normlized_intensity(z)
    
    # fixed for the problem of pickle
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Perform additional initialization if needed

    def __getstate__(self):
        return self.__dict__


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
    def __init__(self, layer:tuple[list[float],list[complex|eps_userdefine]], lock=None, **kwargs):
        import uuid
        if lock is None: lock = mp.Manager().Lock()
        self.layer_thicknesses = np.array(layer[0])
        self.epsilons = np.array(layer[1])
        self.kwargs = kwargs
        self._check_para()
        self.beta = 2*np.pi/np.sqrt(self.cellsize_x*self.cellsize_y)
        self._gen_avg_eps()
        self._define_kwargs()
        self.tmm = TMM(self.layer_thicknesses, self.avg_epsilons, self.beta, self.kwargs.get('k0_init', None))
        self.tmm.find_modes()
        # Generate a unique id for the model parameters.
        self.uuid = uuid.uuid4().hex
        self._save()
        self.lock = lock

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

    def _define_kwargs(self):
        for key, value in self.kwargs.items():
            setattr(self, key, value)

    def _save(self):
        import os
        if not os.path.exists(f'./history_res/{self.uuid}/'):
            os.mkdir(f'./history_res/{self.uuid}/')
            np.save(f'./history_res/{self.uuid}/input_para.npy', self.__dict__)
        else:
            warnings.warn(f'Warning: The folder ./history_res/{self.uuid}/ is already exist. The data will be used to pass the calculation.', FutureWarning)


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
        from calculator import integral_method
        integrated_func_2d = integral_method(3, 'dblquad')()
        integrated_func_1d = integral_method(3, 'quad')()
        self.paras = paras
        self.pathname_suffix = self.paras.uuid
        self.tmm = self.paras.tmm
        self.k0 = self.tmm.k0
        self.beta0 = self.tmm.beta
        self.lock = self.paras.lock
        self.integrated_func_2d = integrated_func_2d
        self.integrated_func_1d = integrated_func_1d
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
        from calculator import xi_calculator
        # Detect the boundary of the eps_userdefine. Assuming photonic crystal layers are continuous.
        self.phc_boundary = []
        self.xi_calculator_collect = []
        for i in range(len(self.paras.epsilons)):
            if isinstance(self.paras.epsilons[i], eps_userdefine):
                self.phc_boundary.append(self.tmm.z_boundary[i])
                self.phc_boundary.append(self.tmm.z_boundary[i+1])
                self.xi_calculator_collect.append(xi_calculator(self.paras.epsilons[i], 'xi(index=(m,n))', self.pathname_suffix, lock=self.lock))
            else:
                self.xi_calculator_collect.append(None)
        self.xi_calculator_collect = np.array(self.xi_calculator_collect)
        self.phc_boundary = np.array(self.phc_boundary)
        self.phc_boundary.sort()
        self.gamma_phc = self.integrated_func_1d(self.tmm.e_normlized_intensity, self.phc_boundary[0], self.phc_boundary[-1])        
        self.coupling_coeff = np.array([self.integrated_func_1d(self.tmm.e_normlized_intensity, self.tmm.z_boundary[i], self.tmm.z_boundary[i+1]) for i in range(len(self.tmm.z_boundary)-1)])
        self._xi_weight = np.array([self.coupling_coeff[_]/self.gamma_phc for _ in range(len(self.coupling_coeff))])

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
        i = self.tmm._find_layer(z)
        xi_calculator_i = self.xi_calculator_collect[i]
        def _process_element(input_element):
            if input_element:
                return input_element[order]
            else:
                return 0+0j
        return np.vectorize(_process_element)(xi_calculator_i)

    def _mu_func(self, order):
        m, n, r, s = order
        def integrated_func(z, z_prime):
            return self.xi_z_func(z_prime,(m-r,n-s))*self.Green_func_higher_order(z,z_prime,(m,n))*self.tmm.e_normlized_amplitude(z_prime)*np.conj(self.tmm.e_normlized_amplitude(z))
        return 1/np.square(self.k0)*self.integrated_func_2d(integrated_func, self.phc_boundary[0], self.phc_boundary[-1], self.phc_boundary[0], self.phc_boundary[-1])
    
    def _nu_func(self, order):
        m, n, r, s = order
        def integrated_func(z):
            return 1/self.__eps_profile_z(z)*self.xi_z_func(z,(m-r,n-s))*self.tmm.e_normlized_intensity(z)
        return -self.integrated_func_1d(integrated_func, self.phc_boundary[0], self.phc_boundary[-1])

    def _zeta_func(self, order):
        print(f'\rzeta: {order}          ', end='', flush=True)
        p, q, r, s = order
        def integrated_func(z, z_prime):
            return self.xi_z_func(z,(p,q))*self.xi_z_func(z,(-r,-s))*self.Green_func_fundamental(z,z_prime)*self.tmm.e_normlized_amplitude(z_prime)*np.conj(self.tmm.e_normlized_amplitude(z))
        return -np.square(np.square(self.k0))/(2*self.beta0)*self.integrated_func_2d(integrated_func, self.phc_boundary[0], self.phc_boundary[-1], self.phc_boundary[0], self.phc_boundary[-1])
    
    def _kappa_func(self, order):
        m, n = order
        def integrated_func(z):
            return self.xi_z_func(z,(m,n))*self.tmm.e_normlized_intensity(z)
        return -np.square(self.k0)/(2*self.beta0)*self.integrated_func_1d(integrated_func, self.phc_boundary[0], self.phc_boundary[-1])


class CWT_solver():
    """
    
    """
    def __init__(self, model:Model):
        self.model = model
        self.lock = self.model.lock
        self.core_num = mp.cpu_count()
        self.prepare_calculator()

    def __getattr__(self, name):
        if name == '_cut_off':
            print('cut_off not set. Use default value 10.')
            self._cut_off = 10
            return self._cut_off
    
    def prepare_calculator(self):
        from calculator import Array_calculator, varsigma_matrix_calculator
        self.xi_calculator_collect = self.model.xi_calculator_collect
        self.varsigma_matrix_calculator = varsigma_matrix_calculator(self.model, notes='varsigma_matrix((m,n))', pathname_suffix=self.model.pathname_suffix, lock=self.lock)
        self.zeta_calculator = Array_calculator(self.model._zeta_func, notes='zeta(index=(p,q,r,s))', pathname_suffix=self.model.pathname_suffix, lock=self.lock)
        self.kappa_calculator = Array_calculator(self.model._kappa_func, notes='kappa(index=(m,n))', pathname_suffix=self.model.pathname_suffix, lock=self.lock)
        self.get_varsigma = self.varsigma_matrix_calculator.get_varsigma

    def _chi_func(self, order, direction:str):
        cut_off = self._cut_off
        p, q, r, s = order
        def sumed_func(input): # TODO: Check the formula. The formulas may be wrong in integration.
            m, n = input
            avg_xi = np.sum([self.xi_calculator_collect[_][p-m,q-n]*self.model._xi_weight[_] for _ in range(len(self.xi_calculator_collect)) if self.xi_calculator_collect[_] is not None])
            return avg_xi*self.get_varsigma((m,n,r,s), direction)
        m_mesh = np.arange(-cut_off, cut_off+1)
        n_mesh = np.arange(-cut_off, cut_off+1)
        MM, NN = np.meshgrid(m_mesh, n_mesh)
        MM, NN = MM.flatten(), NN.flatten()
        iter = [(m,n) for m,n in zip(MM,NN)  if m**2+n**2 > 1]
        result = np.array([sumed_func(i) for i in iter])
        return  np.sum(result)

    def _pre_cal_(self):
        from calculator import Array_calculator
        import time
        t1 = time.time()
        m_mesh = np.arange(-self._cut_off-1, self._cut_off+2)
        n_mesh = np.arange(-self._cut_off-1, self._cut_off+2)
        MM, NN = np.meshgrid(m_mesh, n_mesh)
        MM, NN = MM.flatten(), NN.flatten()
        with mp.Pool(self.core_num) as pool:
            # xi
            iter = [(m,n) for m,n in zip(MM,NN)  if m**2+n**2 >= 1]
            for f in self.xi_calculator_collect:
                if isinstance(f, Array_calculator):
                    res = pool.map(f, iter)
                    f.enable_edit()
                    for i, r in zip(iter, res):
                        f[i] = r
                    f.disable_edit()
        m_mesh = np.arange(-self._cut_off, self._cut_off+1)
        n_mesh = np.arange(-self._cut_off, self._cut_off+1)
        MM, NN = np.meshgrid(m_mesh, n_mesh)
        MM, NN = MM.flatten(), NN.flatten()
        with mp.Pool(self.core_num) as pool:
            # # varsigma
            iter = [(m,n) for m,n in zip(MM,NN)  if m**2+n**2 > 1]
            res = pool.map(self.varsigma_matrix_calculator, iter)
            self.varsigma_matrix_calculator.enable_edit()
            for i, r in zip(iter, res):
                self.varsigma_matrix_calculator[i] = r
            self.varsigma_matrix_calculator.disable_edit()
        with mp.Pool(self.core_num) as pool:
            # zeta
            iter=[(1,0,1,0),(1,0,-1,0),(-1,0,1,0),(-1,0,-1,0),(0,1,0,1),(0,1,0,-1),(0,-1,0,1),(0,-1,0,-1)]
            res = pool.map(self.zeta_calculator, iter)
            self.zeta_calculator.enable_edit()
            for i, r in zip(iter, res):
                self.zeta_calculator[i] = r
            self.zeta_calculator.disable_edit()
        t2 = time.time()
        self._pre_cal_time = t2-t1
        print('\rPre-calculation finished. Time cost: ', self._pre_cal_time, 's.', flush=True)

    def cal_coupling_martix(self, cut_off=10, parallel=True):
        self._cut_off = cut_off
        if parallel: self._pre_cal_()
        kappa = self.kappa_calculator
        zeta = self.zeta_calculator
        chi = self._chi_func
        C_mat_1D = np.array([[0, kappa[2,0], 0, 0],
                              [kappa[-2,0], 0, 0, 0],
                              [0, 0, 0, kappa[0,2]],
                              [0, 0, kappa[0,-2], 0]])
        C_mat_rad = np.array([[zeta[1,0,1,0], zeta[1,0,-1,0], 0, 0],
                              [zeta[-1,0,1,0], zeta[-1,0,-1,0], 0, 0],
                              [0, 0, zeta[0,1,0,1], zeta[0,1,0,-1]],
                              [0, 0, zeta[0,-1,0,1], zeta[0,-1,0,-1]]])
        C_mat_2D = np.array([[chi((1,0,1,0),'y'), chi((1,0,-1,0),'y'), chi((1,0,0,1),'y'), chi((1,0,0,-1),'y')],
                             [chi((-1,0,1,0),'y'), chi((-1,0,-1,0),'y'), chi((-1,0,0,1),'y'), chi((-1,0,0,-1),'y')],
                             [chi((0,1,1,0),'x'), chi((0,1,-1,0),'x'), chi((0,1,0,1),'x'), chi((0,1,0,-1),'x')],
                             [chi((0,-1,1,0),'x'), chi((0,-1,-1,0),'x'), chi((0,-1,0,1),'x'), chi((0,-1,0,-1),'x')]])
        self.C_mats = {'1D':C_mat_1D, 'rad':C_mat_rad, '2D':C_mat_2D}
        self.cal_eign_value()
        self._save()
        print(f'Calculation finished. Results is saved to ./history_res/{self.model.pathname_suffix}_res.npy', flush=True)
    
    def cal_eign_value(self):
        from scipy.linalg import eig
        from scipy.constants import c
        self.c = c*1e6 # um/s
        self.C_mat_sum = np.sum([_ for _ in self.C_mats.values()], axis=0)
        self.eigen_values, self.eigen_vectors = eig(self.C_mat_sum)
        self.delta = np.real(self.eigen_values)
        self.alpha = np.imag(self.eigen_values)
        self.alpha_r = 2*self.alpha
        self.beta0 = self.model.beta0
        self.beta = self.beta0+self.delta
        self.k0 = self.model.k0
        self.omega0 = self.k0*self.c
        self.n_eff = self.beta0/self.k0
        self.omega = self.omega0+self.delta/self.n_eff*self.c
        self.Q = self.beta/self.alpha_r

    def _save(self):
        import os
        if not os.path.exists(f'./history_res/{self.model.pathname_suffix}/'):
            os.mkdir(f'./history_res/{self.model.pathname_suffix}/')
        save_dict = {'C_mats':self.C_mats,
                    'eigen_values':self.eigen_values,
                    'eigen_vectors':self.eigen_vectors,
                    'delta':self.delta,
                    'alpha':self.alpha,
                    'alpha_r':self.alpha_r,
                    'beta0':self.beta0,
                    'beta':self.beta,
                    'k0':self.k0,
                    'omega0':self.omega0,
                    'omega':self.omega,
                    'n_eff':self.n_eff,
                    'Q':self.Q}
        np.save(f'./history_res/{self.model.pathname_suffix}/CWT_res.npy', save_dict)