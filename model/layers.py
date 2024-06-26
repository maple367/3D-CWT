import numpy as np
from scipy.optimize import Bounds, dual_annealing, minimize, direct
from scipy.integrate import quad
import warnings

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

    def _construct_matrix(self, k0):
        self.gamma_s = np.sqrt(np.square(self.beta)-np.square(k0)*self.epsilons)
        prime = self.gamma_s[:-1]*self.layer_thicknesses[:-1]
        coeff_s = self.gamma_s[:-1]/self.gamma_s[1:]
        T_mat_s_00 = (1+coeff_s)*np.exp(prime)/2
        T_mat_s_01 = (1-coeff_s)*np.exp(-prime)/2
        T_mat_s_10 = (1-coeff_s)*np.exp(prime)/2
        T_mat_s_11 = (1+coeff_s)*np.exp(-prime)/2
        self.T_mat_s = np.zeros((len(self.layer_thicknesses)-1,2,2)) # Must define the dtype, otherwise the default dtype is float64, which will cause error in assign complex number to float64.
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
        z_s = self.z_boundary[1:]
        for i in range(len(z_s)):
            if z < z_s[i]:
                break
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
    
    def find_modes(self):
        """
        return: the k0 of the mode
        """
        def t_11_func_k_log(k0):
            self._construct_matrix(k0)
            log_t11 = np.log10(np.abs(self.t_11))
            return log_t11
        def t_11_func_k(k0):
            self._construct_matrix(k0)
            return np.abs(self.t_11)
        if self.k0_init is None:
            k0_min = np.array(np.real(self.beta/np.sqrt(np.real(self.epsilons).max())), dtype=np.longdouble) # k0 must be smaller than wave in highest epsilon medium
            k0_max = np.array(np.real(self.beta/np.sqrt(np.real(self.epsilons)[0])), dtype=np.longdouble) # k0 must be bigger than wave in cladding medium
            k_sol_coarse = direct(t_11_func_k_log, Bounds(k0_min,k0_max))
            k0_init = k_sol_coarse.x
        else:
            k0_init = self.k0_init
        k_sol = minimize(t_11_func_k_log, x0=k0_init, method='Nelder-Mead')
        k_sol = minimize(t_11_func_k, x0=k_sol.x, method='Nelder-Mead')
        while k_sol.fun > -15.0 and self.find_modes_iter < 5:
            self.find_modes_iter += 1
            if k_sol.fun < -10.0:
                self.k0_init = k_sol.x
            warnings.warn(f't11 is larger than 1e-15 after {k_sol.nit} iter, t11 = {k_sol.fun}. Retry {self.find_modes_iter}.', RuntimeWarning)
            self.find_modes()
        self.k0 = k_sol.x
        self._construct_matrix(self.k0)
        self._cal_normlized_constant()

    def _cal_normlized_constant(self):
        e_amp_min_r = minimize(lambda z: np.abs(np.real(self.e_amplitude(z)[0])), x0=self.z_boundary[-1], method='Nelder-Mead')
        e_amp_min_l = minimize(lambda z: np.abs(np.real(self.e_amplitude(z)[0])), x0=self.z_boundary[0], method='Nelder-Mead')
        e_amp_min_boundary_distance = min(np.abs(e_amp_min_r.x-self.z_boundary[-1]), np.abs(e_amp_min_l.x-self.z_boundary[0]))
        self._normlized_bd = e_amp_min_boundary_distance
        print(f'Boundary distance: {self._normlized_bd}')
        e_amp_integral = quad(lambda z: np.square(np.real(self.e_amplitude(z)[0])), self.z_boundary[0]-self._normlized_bd, self.z_boundary[-1]+self._normlized_bd)
        self._normlized_constant = 1/e_amp_integral[0]
        print(f'Normlized constant: {self._normlized_constant}')

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

