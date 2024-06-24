# class for construct matrix for TMM

import numpy as np

class TMM_cal():
    """
    k0: wave number in vacuum
    t_s: thickness of each layer, list or array
    eps_s: epsilon of each layer, list or array
    The layer structure is looked like:

    --|eps1,t0|eps2,t1|eps3,t2|--
    where eps0 and eps4 are the epsilon of the two boundarys, eps1, eps2, eps3 are the epsilon of the three layers, t0, t1, t2 are the thickness of the three layers.
    return: the value needed for constructing matrix of TMM
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

# %%
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

# %%
