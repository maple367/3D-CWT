import numpy as np
import os
import mph
from scipy.optimize import Bounds, dual_annealing, minimize, direct, differential_evolution
from scipy.integrate import quad
import numba
import warnings
from model.rect_lattice import eps_userdefine
from model import material_class
vectorize_isinstance = np.vectorize(isinstance, excluded=['class_or_tuple'])
import multiprocessing as mp
import dill
dill.extend(use_dill=True)

class opt_sol():
    def __init__(self, x_sols, y_sols, thershold=None):
        self.x_sols = x_sols
        self.y_sols = y_sols
        self.thershold = thershold
        if self.thershold is None:
            self.single_sol()
        else:
            self.multi_sol(self.thershold)

    def single_sol(self):
        min_index = np.argmin(self.y_sols)
        self.x = self.x_sols[min_index]
        self.fun = self.y_sols[min_index]

    def multi_sol(self, thershold):
        self.x = self.x_sols[self.y_sols < thershold]
        self.fun = self.y_sols[self.y_sols < thershold]

def singlestart_opt(func, x0, method='TNC'):
    res = minimize(func, x0, method=method)
    return res.x, res.fun
singlestart_opt = np.vectorize(singlestart_opt, excluded=['func', 'method'])
def multistart_opt(func, bounds, grid_num=50, method='TNC', thershold=None):
    x0s = np.linspace(bounds[0], bounds[-1], grid_num)
    x_sols0, y_sols0 = singlestart_opt(func, x0s, method)
    # remove the solutions that are not converged
    y_sols = np.array([y_sols0[i] for i in range(len(y_sols0)) if np.isfinite(y_sols0[i])])
    x_sols = np.array([x_sols0[i] for i in range(len(x_sols0)) if np.isfinite(y_sols0[i])])
    return opt_sol(x_sols, y_sols, thershold)

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
    layer_thicknesses : iterable[float]
        Thickness of each layer.
    epsilons : iterable[float, complex]
        Dielectric constant of each layer.
    k0 : float
        Wave number in free space.
    beta_init : float, complex
        Propagation constant of the fundamental mode.

    Returns
    -------
    out : class callable
        z : the position of the field, the z = 0 is the boundary of the first layer
        return : the E field normlized intensity. $$\\int_{-\\infty}^{\\infty} |E_amp|^2 dz = 1$$
    """
    def __init__(self, layer_thicknesses, epsilons, k0, beta_init=None, surface_grating=False):
        self.find_modes_iter = 0
        self.k0 = k0
        self.beta_init = beta_init
        self.layer_thicknesses = np.array(layer_thicknesses, dtype=np.longdouble)
        self.epsilons = np.array(epsilons, dtype=np.longcomplex)
        self.z_boundary = np.insert(np.cumsum(self.layer_thicknesses), 0, 0)
        self._layer_thicknesses_ = self.layer_thicknesses
        self._epsilons_ = self.epsilons
        self.surface_grating = surface_grating
        if self.surface_grating: # add a air layer at the top of the layer, because the field in surface grating is extent to air.
            self._layer_thicknesses_ = np.insert(self._layer_thicknesses_, 0, 0.0)
            self._epsilons_ = np.insert(self._epsilons_, 0, 1.0+0.0j)
        self.beta = np.array(self.beta_init, dtype=np.longcomplex)
        self._z_boundary_ = np.insert(np.cumsum(self._layer_thicknesses_), 0, 0) # the z position of the boundary of each layer
        self._z_boundary_without_lb = self._z_boundary_[1:]
        self._len_z_boundary = len(self._z_boundary_)
        self.beta_r_sorted = self.k0*np.sort(np.real(np.sqrt(np.unique(self._epsilons_))))
        self.beta_r_max = self.beta_r_sorted[-1] # beta is max in the medium with max epsilon
        self.beta_r_min = self.beta_r_sorted[0] # beta is min in the medium with min epsilon, except air
        self.conveged = False

    def _construct_matrix(self, beta):
        self.gamma_s = np.sqrt(np.square(beta)-np.square(self.k0)*self._epsilons_)
        prime = self.gamma_s[:-1]*self._layer_thicknesses_[:-1]
        coeff_s = self.gamma_s[:-1]/self.gamma_s[1:]
        T_mat_s_00 = (1+coeff_s)*np.exp(prime)/2
        T_mat_s_01 = (1-coeff_s)*np.exp(-prime)/2
        T_mat_s_10 = (1-coeff_s)*np.exp(prime)/2
        T_mat_s_11 = (1+coeff_s)*np.exp(-prime)/2
        self.T_mat_s = np.zeros((len(self._layer_thicknesses_)-1,2,2), dtype=np.longcomplex) # Must define the dtype, otherwise the default dtype is float64, which will cause error in assign complex number to float64.
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
            The index of the layer, with the additional air layer if the surface_grating is True.
        """
        return __find_layer__(z, self._z_boundary_without_lb, self._len_z_boundary-2)
    
    def find_layer(self, z):
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
        if self.surface_grating:
            return self._find_layer(z)-1
        else:
            return self._find_layer(z)
    
    def e_amplitude(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field amplitude
        """
        i = self._find_layer(z)
        V = self.V_s_flatten[i]
        z_b = self._z_boundary_[i]
        V_0 = np.take(V,0,axis=-1)
        V_1 = np.take(V,1,axis=-1)
        e_amp = V_0*np.exp(self.gamma_s[i]*(z-z_b))+V_1*np.exp(-self.gamma_s[i]*(z-z_b))
        eps = self._epsilons_[i]
        return e_amp, eps
    
    def e_normlized_intensity(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field normlized intensity. $$\\int_{-\\infty}^{\\infty} |E_amp|^2 dz = 1$$
        """
        e_amp, _ = self.e_amplitude(z)
        e_intensity = np.square(np.abs(e_amp))*self._normlized_constant
        return e_intensity
    
    def e_normlized_amplitude(self, z):
        """
        z: the position of the field, the z = 0 is the boundary of the first layer
        return: the E field normlized amplitude. $$\\int_{-\\infty}^{\\infty} |E_amp|^2 dz = 1$$
        """
        e_amp, _ = self.e_amplitude(z)
        e_norm_amp = e_amp*np.sqrt(self._normlized_constant)
        return e_norm_amp
    
    def find_modes(self):
        """
        return: the k0 of the mode
        """
        def t_11_func_beta_log(beta:tuple[float, float]):
            beta = beta[0] + 1j*beta[1]
            self._construct_matrix(beta)
            log_t11 = np.log10(np.abs(self.t_11))
            return log_t11
        def t_11_func_beta_log_real(beta:float):
            beta = beta + 1j*0.0
            self._construct_matrix(beta)
            log_t11 = np.log10(np.abs(self.t_11))
            return log_t11
        beta_sol_fun = 1.0
        if self.conveged: beta_sol = self.beta_sol
        while (beta_sol_fun >= -10.0) and (self.find_modes_iter < len(self.beta_r_sorted)-1):
            if self.beta_init is None:
                beta_sol = multistart_opt(t_11_func_beta_log_real, bounds=[self.beta_r_min, self.beta_r_max], grid_num=50*(self.find_modes_iter+1), method='TNC')
                beta_init = np.array([beta_sol.x, 0])
            else:
                beta_init = self.beta_init
            beta_sol = minimize(t_11_func_beta_log, x0=beta_init, method='Nelder-Mead')
            if beta_sol.fun < -6.0:
                self.beta_init = beta_sol.x
                self.conveged = True
            beta_sol_fun = beta_sol.fun
            self.find_modes_iter += 1
        self.beta = beta_sol.x[0]+1j*beta_sol.x[1]
        self.beta_sol = beta_sol
        _ = t_11_func_beta_log(beta_sol.x) # update the T_total, V_s, V_s_flatten
        # if _ > -10.0: raise RuntimeError(f't11 is not converge! abs(t11) = {10**_}.')
        self._cal_normlized_constant()
        self.V_s_flatten = np.array([_.flatten() for _ in self.V_s])

    def _cal_normlized_constant(self):
        def e_amp_real_func(z):
            return np.abs(np.real(self.e_amplitude(z)[0]))
        def e_amp_square_func(z):
            return np.square(np.abs(self.e_amplitude(z)[0]))
        e_amp_min_r = minimize(e_amp_real_func, x0=self._z_boundary_[-1], method='Nelder-Mead')
        e_amp_min_l = minimize(e_amp_real_func, x0=self._z_boundary_[0], method='Nelder-Mead')
        e_amp_min_boundary_distance = min(np.abs(e_amp_min_r.x-self._z_boundary_[-1]), np.abs(e_amp_min_l.x-self._z_boundary_[0]))
        self._normlized_bd = e_amp_min_boundary_distance
        e_amp_integral = quad(e_amp_square_func, self._z_boundary_[0]-self._normlized_bd, self._z_boundary_[-1]+self._normlized_bd)
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
        return: the E field normlized intensity. $$\\int_{-\\infty}^{\\infty} |E_amp|^2 dz = 1$$
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
    **kwargs : keyword arguments
        The other parameters of the model.

    Returns
    -------
    out : class
        The model parameters.
    """
    def __init__(self, input_para:tuple[list[float],list[material_class|eps_userdefine],dict[list,list]]=None, load_path=None, **kwargs):
        if load_path is not None:
            self._load(load_path)
        else:
            self._init(input_para, **kwargs)
    
    def _init(self, input_para, **kwargs):
        import uuid
        import time
        self.layer_thicknesses = np.array(input_para[0])
        self.materials = input_para[1]
        self.doping_para = input_para[2]
        self._check_para()
        self.k0 = kwargs.get('k0', 2*np.pi/0.98) # 0.98um is the default wavelength of the light source
        self.beta_init = kwargs.get('beta_init', None)
        self.surface_grating = kwargs.get('surface_grating', False)
        self._gen_avg_eps()
        self.tmm = TMM(self.layer_thicknesses, self.avg_epsilons, self.k0, self.beta_init, self.surface_grating)
        self.tmm.find_modes()
        self._update_cellsize_()
        # Generate a unique id for the model parameters.
        self.uuid = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))+'_'+uuid.uuid4().hex
        self._save()
        self.lock = mp.Manager().Lock()

    def _check_para(self):
        num_layer_phc = 0
        # cellsize_x = []
        # cellsize_y = []
        self.epsilons = []
        for _ in self.materials:
            if isinstance(_, eps_userdefine):
                num_layer_phc += 1
                # cellsize_y.append(_.cell_size_y)
                self.epsilons.append(_)
            elif isinstance(_, material_class):
                self.epsilons.append(_.epsilon)
            else:
                raise ValueError('The type of material must be eps_userdefine or material_class.')
        if num_layer_phc == 0: raise ValueError("At least one eps_userdefine is needed in epsilons.")
        # cellsize is automatically determined by the model.TMM now. Ref to _update_cellsize_.
        # assert len(set(cellsize_x)) == 1, 'The cellsize_x of eps_userdefine must be the same.'
        # assert len(set(cellsize_y)) == 1, 'The cellsize_y of eps_userdefine must be the same.'
        # self.cellsize_x = cellsize_x[0]
        # self.cellsize_y = cellsize_y[0]
        self.epsilons = np.array(self.epsilons)

    def _gen_avg_eps(self):
        avg_eps = []
        for _ in self.epsilons:
            if isinstance(_, eps_userdefine):
                avg_eps.append(_.avg_eps)
            else:
                avg_eps.append(_)
        self.avg_epsilons = np.array(avg_eps)

    def _update_cellsize_(self):
        a = 2*np.pi/np.real(self.tmm.beta)
        print(f'a: {a} um')
        self.cellsize_x = a
        self.cellsize_y = a
        for i in range(len(self.materials)):
            if isinstance(self.materials[i], eps_userdefine):
                self.materials[i].cell_size_x = a
                self.materials[i].cell_size_y = a
                self.materials[i].build()

    def _save(self):
        import os
        if not os.path.exists(f'./history_res/'):
            os.mkdir(f'./history_res/')
        if not os.path.exists(f'./history_res/{self.uuid}/'):
            os.mkdir(f'./history_res/{self.uuid}/')
            np.save(f'./history_res/{self.uuid}/input_para.npy', self.__dict__)
        else:
            warnings.warn(f'Warning: The folder ./history_res/{self.uuid}/ is already exist. The data will be used to pass the calculation.', FutureWarning)
            self._load(f'./history_res/{self.uuid}/input_para.npy')
        print(f'The model parameters is saved in ./history_res/{self.uuid}/input_para.npy.')

    def _load(self, path):
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} is not found.')
        self.__dict__.update(np.load(path, allow_pickle=True).item())
        self.lock = mp.Manager().Lock()


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
    def __init__(self, paras:model_parameters, fast_mode=False):
        from calculator import integral_method
        integrated_func_2d = integral_method(3, 'dblquad')()
        integrated_func_1d = integral_method(3, 'quad')()
        self.paras = paras
        self.lock = self.paras.lock
        self.pathname_suffix = self.paras.uuid
        self.tmm = self.paras.tmm
        self.z_boundary = self.tmm.z_boundary
        self.find_layer = self.tmm.find_layer
        self.e_normlized_intensity = self.tmm.e_normlized_intensity
        self.e_normlized_amplitude = self.tmm.e_normlized_amplitude
        self.k0 = self.tmm.k0
        self.beta0 = self.tmm.beta
        self.__no_doping_min__, self.__no_doping_max__ = np.min(np.where(np.array(self.paras.doping_para['is_no_doping']) == True)), np.max(np.where(np.array(self.paras.doping_para['is_no_doping']) == True))
        self.integrated_func_2d = integrated_func_2d
        self.integrated_func_1d = integrated_func_1d
        self.prepare_calculator(fast_mode)
    
    def _doping_(self, z):
        if z < self.z_boundary[0]:
            z = self.z_boundary[0]
        elif z > self.z_boundary[-1]:
            z = self.z_boundary[-1]
        num_layer = self.find_layer(z)
        if self.paras.doping_para['is_no_doping'][num_layer]:
            return 0
        elif num_layer < self.__no_doping_min__:
            return np.exp(self.paras.doping_para['coeff'][0] + self.paras.doping_para['coeff'][1]*z)
        elif num_layer > self.__no_doping_max__:
            return np.exp(self.paras.doping_para['coeff'][2] + self.paras.doping_para['coeff'][3]*z)
        else:
            raise ValueError('The z corrdinate is unexpected.')
    
    def doping(self, z):
        return np.vectorize(self._doping_)(z)
    
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
        num_layer = self.find_layer(z)
        eps = self.paras.epsilons[num_layer]
        is_eps_userdefine = vectorize_isinstance(eps, eps_userdefine)
        eps_array = np.empty_like(z, dtype=object)
        for i in range(len(z)):
            if is_eps_userdefine[i]:
                eps_array[i] = eps[i].eps(x, y)
            else:
                eps_array[i] = np.ones_like(x)*eps[i]
        return eps_array
    
    def __eps_profile_z(self, z):
        num_layer = self.find_layer(z)
        return self.paras.avg_epsilons[num_layer]
    
    def is_in_phc(self, z):
        def _is_in_phc(z):
            num_layer = self.find_layer(z)
            if isinstance(self.paras.epsilons[num_layer], eps_userdefine):
                return True
            else:
                return False
        _is_in_phc = np.vectorize(_is_in_phc)
        return _is_in_phc(z)  

    def __call__(self, x=None, y=None, z=None):
        return self.eps_profile(x, y, z)
    
    def prepare_calculator(self, fast_mode):
        from calculator import xi_calculator, xi_calculator_DFT
        # Detect the boundary of the eps_userdefine. Assuming photonic crystal layers are continuous.
        self.phc_boundary_l = []
        self.phc_boundary_r = []
        self.xi_calculator_collect = []
        for i in range(len(self.paras.epsilons)):
            if isinstance(self.paras.epsilons[i], eps_userdefine):
                self.phc_boundary_l.append(self.z_boundary[i])
                self.phc_boundary_r.append(self.z_boundary[i+1])
                if fast_mode: self.xi_calculator_collect.append(xi_calculator_DFT(self.paras.epsilons[i], f'xi((m,n))[{i}]', self.pathname_suffix, lock=self.lock))
                else: self.xi_calculator_collect.append(xi_calculator(self.paras.epsilons[i], f'xi((m,n))[{i}]', self.pathname_suffix, lock=self.lock))
            else:
                self.xi_calculator_collect.append(None)
        self._generate_integral_region_()
        self.xi_calculator_collect = np.array(self.xi_calculator_collect)
        self.gamma_phc = np.sum([ [self.integrated_func_1d(self.e_normlized_intensity, bd[0], bd[1]) for bd in self._1d_phc_integral_region_]])    
        self.coupling_coeff = np.array([self.integrated_func_1d(self.e_normlized_intensity, self.z_boundary[i], self.z_boundary[i+1]) for i in range(len(self.z_boundary)-1)])
        self._xi_weight = np.array([self.coupling_coeff[_]/self.gamma_phc for _ in range(len(self.coupling_coeff))])
        self._fc_coupling_p_ = self.integrated_func_1d(lambda z: self.e_normlized_intensity(z)*self.doping(z), self.z_boundary[0], self.z_boundary[self.__no_doping_min__])
        self._fc_coupling_n_ = self.integrated_func_1d(lambda z: self.e_normlized_intensity(z)*self.doping(z), self.z_boundary[self.__no_doping_max__+1], self.z_boundary[-1])
        self.fc_absorption = self._fc_coupling_p_*7e-10+self._fc_coupling_n_*3e-10
        self.kappa_v = -(self.k0**4)/(2*self.beta0)*np.sum([self.integrated_func_2d(lambda z,z_prime: self.Green_func_fundamental(z,z_prime)*self.e_normlized_amplitude(z_prime)*np.conj(self.e_normlized_amplitude(z)), bd[0], bd[1], bd[2], bd[3]) for bd in self._2d_phc_integral_region_])

    def _generate_integral_region_(self):
        self._1d_phc_integral_region_ = []
        for i in range(len(self.phc_boundary_l)):
            self._1d_phc_integral_region_.append([self.phc_boundary_l[i], self.phc_boundary_r[i]])
        self._2d_phc_integral_region_ = []
        for i in range(len(self.phc_boundary_l)):
            for j in range(len(self.phc_boundary_l)):
                self._2d_phc_integral_region_.append([self.phc_boundary_l[i], self.phc_boundary_r[i], self.phc_boundary_l[j], self.phc_boundary_r[j]])

    def Green_func_fundamental(self, z, z_prime):
        # Approximatly Green function
        return -1j/(2*self.beta_z_func_fundamental(z))*np.exp(-1j*self.beta_z_func_fundamental(z)*np.abs(z-z_prime))
    
    def beta_z_func_fundamental(self, z):
        return self.k0*np.sqrt(self.__eps_profile_z(z))
    
    def Green_func_higher_order(self, z, z_prime, order):
        # Approximatly Green function of higher order
        return 1/(2*self.beta_z_func_higher_order(z, order))*np.exp(-self.beta_z_func_higher_order(z, order)*np.abs(z-z_prime))
    
    def beta_z_func_higher_order(self, z, order):
        m, n = order
        return np.sqrt( (np.square(m)+np.square(n))*np.square(self.beta0) - np.square(self.k0)*self.__eps_profile_z(z) )
    
    def xi_z_func(self, z, order):
        i = self.find_layer(z)
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
            return self.xi_z_func(z,(m-r,n-s))*self.Green_func_higher_order(z,z_prime,(m,n))*self.e_normlized_amplitude(z_prime)*np.conj(self.e_normlized_amplitude(z))
        res = [self.integrated_func_2d(integrated_func, bd[0], bd[1], bd[2], bd[3]) for bd in self._2d_phc_integral_region_]
        return np.square(self.k0)*np.sum(res)
    
    def _nu_func(self, order):
        m, n, r, s = order
        def integrated_func(z):
            return 1/self.__eps_profile_z(z)*self.xi_z_func(z,(m-r,n-s))*self.e_normlized_intensity(z)
        res = [self.integrated_func_1d(integrated_func, bd[0], bd[1]) for bd in self._1d_phc_integral_region_]
        return -np.sum(res)

    def _zeta_func(self, order):
        print(f'\rzeta: {order}          ', end='', flush=True)
        p, q, r, s = order
        def integrated_func(z, z_prime):
            return self.xi_z_func(z,(p,q))*self.xi_z_func(z,(-r,-s))*self.Green_func_fundamental(z,z_prime)*self.e_normlized_amplitude(z_prime)*np.conj(self.e_normlized_amplitude(z))
        res = [self.integrated_func_2d(integrated_func, bd[0], bd[1], bd[2], bd[3]) for bd in self._2d_phc_integral_region_]
        return -np.square(np.square(self.k0))/(2*self.beta0)*np.sum(res)
    
    def _kappa_func(self, order):
        m, n = order
        def integrated_func(z):
            return self.xi_z_func(z,(m,n))*self.e_normlized_intensity(z)
        res = [self.integrated_func_1d(integrated_func, bd[0], bd[1]) for bd in self._1d_phc_integral_region_]
        return -np.square(self.k0)/(2*self.beta0)*np.sum(res)
    
    def plot(self):
        import matplotlib.pyplot as plt
        z_mesh = np.linspace(self.z_boundary[0], self.z_boundary[-1], 5000)
        E_profile_s = self.e_normlized_intensity(z=z_mesh)
        dopings = self.doping(z=z_mesh)
        eps_s = self.eps_profile(z=z_mesh)
        E_profile_s = E_profile_s / np.max(np.abs(E_profile_s)) * (np.max(np.abs(self.paras.avg_epsilons)) - np.min(np.abs(self.paras.avg_epsilons))) + np.min(np.abs(self.paras.avg_epsilons))
        a_const = self.paras.cellsize_x
        x_mesh = np.linspace(0, a_const, 500)
        y_mesh = np.linspace(0, a_const, 500)
        z_points = np.array([(self.phc_boundary_l[-1]+self.phc_boundary_r[-1])/2,]) # must be a vector
        XX, YY = np.meshgrid(x_mesh, y_mesh)
        eps_mesh_phc = self.eps_profile(XX, YY, z_points)[0]
        color1, color2, fontsize1, fontsize2, fontname = 'mediumblue', 'firebrick', 13, 18, 'serif'
        fig, ax0 = plt.subplots(figsize=(7,5))
        fig.subplots_adjust(left=0.12, right=0.86)
        ax1 = plt.twinx()
        ax0.plot(z_mesh, dopings, color=color1)
        ax0.tick_params(axis='y', colors=color1, labelsize=10)
        ax1.plot(z_mesh, eps_s, linestyle='--', color=color2)
        ax1.plot(z_mesh, E_profile_s, linestyle='--')
        ax1.fill_between(z_mesh, np.min(E_profile_s), E_profile_s, where=self.is_in_phc(z_mesh), alpha=0.4, hatch='//', color='orange')
        ax1.tick_params(axis='y', colors=color2, labelsize=10)
        ax0.set_xlabel(r'z ($\mu m$)', fontsize=fontsize1, fontname=fontname)
        ax0.set_ylabel(r'Doping ($\mu m^{-3}$)', fontsize=fontsize1, fontname=fontname, color=color1)
        ax0.set_yscale('symlog', linthresh=np.min(dopings[dopings!=0.0]))
        ax1.set_ylabel(r'$\epsilon_r$ and Normalized $|E|^2$', fontsize=fontsize1, fontname=fontname, color=color2)
        plt.title('', fontsize=fontsize2, fontname=fontname)
        ax2 = ax0.inset_axes([0.65, 0.10, 0.24, 0.24])
        im = ax2.imshow(np.real(eps_mesh_phc), cmap='Greys')
        ax2.set_xticks([])
        ax2.set_yticks([])
        cb = fig.colorbar(im, cax=ax2.inset_axes([0, 1.05, 1, 0.2]), orientation='horizontal', label='Epsilon')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        plt.show()
        plt.close()


class general_solver():
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        pass

    def _save(self):
        pass


class CWT_solver(general_solver):
    # TODO: Add comments
    """
    
    """
    def __init__(self, model:Model):
        self.model = model
        self.lock = self.model.lock
        self.core_num = min(mp.cpu_count(), 60)
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
        self.kappa_calculator = Array_calculator(self.model._kappa_func, notes='kappa((m,n))', pathname_suffix=self.model.pathname_suffix, lock=self.lock)
        self.get_varsigma = self.varsigma_matrix_calculator.get_varsigma

    def _chi_func(self, order, direction:str):
        cut_off = self._cut_off
        p, q, r, s = order
        def sumed_func(input):
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
        with mp.Pool(self.core_num) as pool:
            m_mesh = np.arange(-self._cut_off-1, self._cut_off+2)
            n_mesh = np.arange(-self._cut_off-1, self._cut_off+2)
            MM, NN = np.meshgrid(m_mesh, n_mesh)
            MM, NN = MM.flatten(), NN.flatten()
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
            # # varsigma
            iter = [(m,n) for m,n in zip(MM,NN)  if m**2+n**2 > 1]
            res = pool.map(self.varsigma_matrix_calculator, iter)
            self.varsigma_matrix_calculator.enable_edit()
            for i, r in zip(iter, res):
                self.varsigma_matrix_calculator[i] = r
            self.varsigma_matrix_calculator.disable_edit()
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

    def run(self, cut_off=10, parallel=True):
        print(f'Start calculation of CWT with cut off {cut_off}...', flush=True)
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
        print(f'Calculation finished. Results is saved to ./history_res/{self.model.pathname_suffix}/CWT_res.npy', flush=True)

    def cal_eign_value(self):
        from scipy.constants import c
        self.c = c*1e6 # um/s
        self.C_mat_sum = np.sum([_ for _ in self.C_mats.values()], axis=0)
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.C_mat_sum)
        self.delta = np.real(self.eigen_values)
        self.alpha = np.imag(self.eigen_values)
        self.alpha_r = 2*self.alpha
        self.beta0 = self.model.beta0
        self.beta = self.beta0+self.delta+self.alpha*1j
        self.k0 = self.model.k0
        self.omega0 = self.k0*self.c
        self.n_eff = self.beta0/self.k0
        self.omega = self.omega0+self.delta/self.n_eff*self.c
        self.k = self.k0+self.delta/self.n_eff
        self.a = self.model.paras.cellsize_x
        self.kappa_v = self.model.kappa_v
        self.xi_rads = self.cal_xi_rads_high_order((0,0))
        self.norm_freq = self.omega/(2*np.pi)/(self.c/self.a)
        self.Q = np.real(self.beta)/self.alpha_r

    def cal_xi_rads_high_order(self, order):
        m, n = order
        xi_rads = np.array([np.sum([self.xi_calculator_collect[_][m-1,n]*self.model._xi_weight[_] for _ in range(len(self.xi_calculator_collect)) if self.xi_calculator_collect[_] is not None]),
                            np.sum([self.xi_calculator_collect[_][m+1,n]*self.model._xi_weight[_] for _ in range(len(self.xi_calculator_collect)) if self.xi_calculator_collect[_] is not None]),
                            np.sum([self.xi_calculator_collect[_][m,n-1]*self.model._xi_weight[_] for _ in range(len(self.xi_calculator_collect)) if self.xi_calculator_collect[_] is not None]),
                            np.sum([self.xi_calculator_collect[_][m,n+1]*self.model._xi_weight[_] for _ in range(len(self.xi_calculator_collect)) if self.xi_calculator_collect[_] is not None])])
        return xi_rads

    def _save(self):
        import os
        if not os.path.exists(f'./history_res/{self.model.pathname_suffix}/'):
            os.mkdir(f'./history_res/{self.model.pathname_suffix}/')
        self.save_dict = {'C_mats':self.C_mats,
                    'C_mat_sum':self.C_mat_sum,
                    'eigen_values':self.eigen_values,
                    'eigen_vectors':self.eigen_vectors,
                    'delta':self.delta,
                    'alpha':self.alpha,
                    'alpha_r':self.alpha_r,
                    'beta0':self.beta0,
                    'beta':self.beta,
                    'k0':self.k0,
                    'k':self.k,
                    'a':self.a,
                    'omega0':self.omega0,
                    'omega':self.omega,
                    'kappa_v':self.kappa_v,
                    'xi_rads':self.xi_rads,
                    'norm_freq':self.norm_freq,
                    'n_eff':self.n_eff,
                    'Q':self.Q}
        np.save(f'./history_res/{self.model.pathname_suffix}/CWT_res.npy', self.save_dict)


class SEMI_solver(general_solver):
    """
    Parameters
    ----------
    comsol_model : mph.client.model
        The comsol model.

    Returns
    -------
    class callable
        method
            run: run the calculation.
                run(class[model])
    """
    def __init__(self, client:mph.Client):
        comsol_model_file_path = os.path.join(os.path.dirname(__file__), '../utils/comsol_model/DQW-1D_20240709.mph')
        self.client = client
        self.comsol_model = self.client.load(comsol_model_file_path)

    def _prepare_model_(self):
        self.doping_para = self.model.paras.doping_para
        self.z_boundary = self.model.z_boundary
        self.__no_doping_min__ = self.model.__no_doping_min__
        self.__no_doping_max__ = self.model.__no_doping_max__
        self.t_p = self.z_boundary[self.__no_doping_min__]-self.z_boundary[0]
        self.t_n = self.z_boundary[-1]-self.z_boundary[self.__no_doping_max__+1]
        self.t_i_p = self.z_boundary[self.__no_doping_min__+2]-self.z_boundary[self.__no_doping_min__]
        self.t_i_n = self.z_boundary[self.__no_doping_max__+1]-self.z_boundary[self.__no_doping_max__]
        self.x_fraction_expr = ''
        for _ in range(len(self.model.paras.materials)):
            if isinstance(self.model.paras.materials[_], eps_userdefine):
                x_fraction = self.model.paras.materials[_].mat_bulk.x
            elif isinstance(self.model.paras.materials[_], material_class):
                x_fraction = self.model.paras.materials[_].x
            else:
                raise ValueError('The type of material must be eps_userdefine or material_class.')
            bd = self.z_boundary[_+1]
            expr_add = f'if(x<={bd},{x_fraction},'
            self.x_fraction_expr += expr_add
        self.x_fraction_expr += '0.0' + ')'*(_+1)
        self.coeff_a = self.doping_para['coeff'][0]
        self.coeff_b = self.doping_para['coeff'][1]
        self.coeff_c = self.doping_para['coeff'][2]
        self.coeff_d = self.doping_para['coeff'][3]

    def _apply_paras_(self):
        self.comsol_model.parameter('t_p',f'{self.t_p}[um]')
        self.comsol_model.parameter('t_i_p',f'{self.t_i_p}[um]')
        self.comsol_model.parameter('t_i_n',f'{self.t_i_n}[um]')
        self.comsol_model.parameter('t_n',f'{self.t_n}[um]')
        x_fraction = self.comsol_model/'functions'/'x_fraction'
        x_fraction.property('expr', self.x_fraction_expr)
        self.comsol_model.parameter('coeff_a',f'{self.coeff_a}')
        self.comsol_model.parameter('coeff_b',f'{self.coeff_b}')
        self.comsol_model.parameter('coeff_c',f'{self.coeff_c}')
        self.comsol_model.parameter('coeff_d',f'{self.coeff_d}')

    def run(self, model:Model):
        print('Start to run SEMI calculation in COMSOL...', flush=True)
        from scipy.constants import e
        self.model = model
        self._prepare_model_()
        self._apply_paras_()
        self.comsol_model.solve()
        self.V0_1 = self.comsol_model.evaluate('semi.V0_1') # V
        self.J0_1 = self.comsol_model.evaluate('semi.J0_1') # A/m^2
        self.R_spon = self.comsol_model.evaluate('intop1(semi.ot1.R_spon)') # 1/m^2
        self.P_spon = self.comsol_model.evaluate('intop1(semi.ot1.P_spon)') # W/m^2
        self.PCE = self.P_spon/(self.V0_1*self.J0_1)
        self.SE = self.R_spon/(self.J0_1/e)
        self._save()
        print('The SEMI calculation is finished.', flush=True)

    def get_result(self,index):
        return self.PCE[index], self.SE[index]
    
    def _save(self):
        import os
        if not os.path.exists(f'./history_res/{self.model.pathname_suffix}/'):
            os.mkdir(f'./history_res/{self.model.pathname_suffix}/')
        self.save_dict = {'V0_1':self.V0_1,
                    'J0_1':self.J0_1,
                    'R_spon':self.R_spon,
                    'P_spon':self.P_spon,
                    'PCE':self.PCE,
                    'SE':self.SE}
        np.save(f'./history_res/{self.model.pathname_suffix}/SEMI_res.npy', self.save_dict)
    

class SGM_solver(general_solver):
    def __init__(self, client:mph.Client):
        comsol_model_file_path = os.path.join(os.path.dirname(__file__), '../utils/comsol_model/cwt_solver_20240922.mph')
        self.client = client
        self.comsol_model = self.client.load(comsol_model_file_path)

    def run(self, res:dict, init_eig_guess:complex, size:int, resolution:int):
        print('Start to run SGM calculation in COMSOL... ', flush=True)
        self.res = res
        self.C_mat_sum = self.res['C_mat_sum']
        self.a = self.res['a']
        self.init_eig_guess = init_eig_guess
        self.size = size
        self.resolution = resolution
        self.kappa_v_i = np.imag(res['kappa_v'])
        self.xi_rads = self.res['xi_rads']
        self._prepare_model_()
        self._apply_paras_()
        self.comsol_model.solve()
        self.eigen_values = self.comsol_model.evaluate('lambda')
        self.P_stim = self.comsol_model.evaluate('2*imag(lambda)*intall(abs(Rx)^2+abs(Sx)^2+abs(Ry)^2+abs(Sy)^2)')
        self.P_edge = self.comsol_model.evaluate('intyd(abs(Sx)^2)+intyd(abs(Rx)^2)+intxd(abs(Sy)^2)+intxd(abs(Ry)^2)') #TODO: checked
        self.P_rad = self.comsol_model.evaluate(f'2*{self.kappa_v_i}*intall(abs({self.xi_rads[0]}*Rx+{self.xi_rads[1]}*Sx)^2+abs({self.xi_rads[2]}*Ry+{self.xi_rads[3]}*Sy)^2)')
        self._save()
        print('The SGM calculation is finished.', flush=True)
    
    def _prepare_model_(self):
        self.coeff_a = [f'{np.real(_)}+{np.imag(_)}j' for _ in self.C_mat_sum.flatten()]

    def _apply_paras_(self):
        coeff = self.comsol_model/'physics'/'系数形式偏微分方程'/'系数形式偏微分方程 1'
        coeff.property('a', self.coeff_a)
        self.comsol_model.parameter('L', f'{self.size}*a')
        self.comsol_model.parameter('a', f'{self.a}')
        self.comsol_model.parameter('resolution', f'{self.resolution}')
        self.comsol_model.parameter('init_eig_guess', f'{np.real(self.init_eig_guess)}+{np.imag(self.init_eig_guess)}j')

    def _preview_fig_(self, i_eigs):
        import matplotlib.pyplot as plt
        pic = self.comsol_model/'plots'/'I'
        pic.property('solnum', i_eigs+1)
        self.comsol_model.export('图像 1','./cwt_res.png')
        fig, ax = plt.subplots()
        ax.imshow(plt.imread(os.path.join(os.path.dirname(__file__), '../utils/comsol_model/cwt_res.png')))
        ax.axis('off')
        return fig
    
    def _get_data_(self, i_eigs):
        import pandas as pd
        data = self.comsol_model/'exports'/'数据 1'
        data.property('solnum', i_eigs+1)
        self.comsol_model.export('数据 1','./cwt_res.csv')
        data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../utils/comsol_model/cwt_res.csv'), skiprows=8)
        return data
    
    def _save(self):
        import os
        if not os.path.exists(f'./history_res/{self.model.pathname_suffix}/'):
            os.mkdir(f'./history_res/{self.model.pathname_suffix}/')
        self.save_dict = {'eigen_values':self.eigen_values,
                    'P_stim':self.P_stim,
                    'P_edge':self.P_edge,
                    'P_rad':self.P_rad}
        np.save(f'./history_res/{self.model.pathname_suffix}/SGM_res.npy', self.save_dict)