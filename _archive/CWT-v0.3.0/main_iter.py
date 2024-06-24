# %%
# What we need to do:
# 1. Define the layer structure
# 2. Define the location and shape of the air holes
# 3. Define the mode profile
# %%
import numpy as np
import scipy.integrate as si
import matlab.engine
eng_m = matlab.engine.start_matlab()

import util

# %%
a = 0.295
beta_0 = 2*np.pi/a
order_lst = np.arange(1,15,1) # order of the Fourier series expansion
freq_lst = []
alpha_lst = []

for order in order_lst:
    print(f'########## order: {order} ##########')
    FF = 0.16
    r = np.sqrt(FF*2)*a
    z_lst_raw = np.array([1.5,0.0885,0.1180,0.0590,1.5])
    eps_lst_raw = np.array([11.0224,12.8603,FF+(1-FF)*12.7449,12.7449,11.0224])
    k0 = util.find_k0(beta_0, z_lst_raw, eps_lst_raw)[0]
    n_eff = beta_0/k0
    z_phc_min = 1.5 + 0.0885
    z_phc_max = 1.5 + 0.0885 + 0.1180

    eps_lst = np.array([11.0224, 12.8603, 12.8603, FF+(1-FF)*12.7449, FF+(1-FF)*12.7449, 12.7449, 12.7449, 11.022])
    z_lst = np.cumsum(z_lst_raw)[:-1].tolist()+(np.cumsum(z_lst_raw)[:-1]+1e-4).tolist()
    z_lst = np.round(z_lst,4)
    z_lst.sort()


    z_mesh_raw = np.linspace(0, np.sum(z_lst_raw), int(np.sum(z_lst_raw)//1e-3+1))
    E_profile_raw = util.TMM_cal(k0, z_lst_raw, eps_lst_raw, beta_0).E_field(z_mesh_raw)[0]
    intensity_integral = si.trapz(np.abs(E_profile_raw)**2, z_mesh_raw)
    E_profile_norm = E_profile_raw/np.sqrt(intensity_integral)

    def E_profile(z):
        return eng_m.E_profile(z, z_mesh_raw, E_profile_norm)
    gamma_phc = si.quad(lambda z: np.square(np.abs(E_profile(z))), z_phc_min, z_phc_max)[0]

    def n0_func(z):
        return eng_m.n0_func(z, z_lst, eps_lst)

    def beta_z_func(z):
        return k0*n0_func(z)

    def G_func(z, zp):
        return -1j/(2*beta_z_func(z))*np.exp(-1j*beta_z_func(z)*np.abs(z-zp))

    ### high order part ###
    def beta_z_func_h(z, m, n):
        return np.sqrt((m**2 + n**2)*beta_0**2 - k0**2*n0_func(z)**2)

    def G_func_h(z, zp, m, n):
        return -1j/(2*beta_z_func_h(z, m, n))*np.exp(-1j*beta_z_func_h(z, m, n)*np.abs(z-zp))
    ### high order part end ###

    # TODO: replace eps_i with variable
    def xi_func(MM, NN):
        xi = eng_m.xi_func(MM.flatten(), NN.flatten(), 12.7449, 1.0, r, a, beta_0)
        xi = xi.reshape(MM.shape)
        return np.array(xi, dtype=np.complex128)

    m_mesh = np.linspace(-(order+1), (order+1), 2*(order+1)+1)
    n_mesh = np.linspace(-(order+1), (order+1), 2*(order+1)+1)
    MM, NN = np.meshgrid(m_mesh, n_mesh, indexing='ij')
    xi_array = xi_func(MM, NN)

    def xi_mat(m: int, n: int, xi_array=xi_array):
        x = xi_array.shape[0]//2+int(m)
        y = xi_array.shape[1]//2+int(n)
        if x < 0 or y < 0 or x >= xi_array.shape[0] or y >= xi_array.shape[1]:
            raise ValueError(f'xi_mat {x},{y} out of range')
        return xi_array[x, y]
    xi_mat = np.vectorize(xi_mat, excluded=['xi_array'])

    def kappa_func(m, n):
        kappa = -k0**2/(2*beta_0)*xi_mat(m, n)*gamma_phc
        return kappa

    def zeta_func(p, q, r, s):
        xi_mat_pq = xi_mat(p, q)
        xi_mat_rs = xi_mat(-r, -s)
        zeta = eng_m.zeta_func(xi_mat_pq, xi_mat_rs, z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, k0, beta_0, np.array(z_lst), np.array(eps_lst))
        return zeta
    
    # chi part
    m_mesh = np.linspace(-(order), (order), 2*(order)+1)
    n_mesh = np.linspace(-(order), (order), 2*(order)+1)
    MM, NN = np.meshgrid(m_mesh, n_mesh, indexing='ij')

    def mu_func(MM, NN, r, s):
        xi_mat_mr_ns = xi_mat(MM.flatten()-r, NN.flatten()-s)
        mu = eng_m.mu_func(MM.flatten(), NN.flatten(), xi_mat_mr_ns, z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, k0, beta_0, z_lst, eps_lst)
        mu = mu.reshape(MM.shape)
        return np.array(mu, dtype=np.complex128)

    mu_array1 = mu_func(MM, NN, 1.0, 0.0)
    mu_array2 = mu_func(MM, NN, -1.0, 0.0)
    mu_array3 = mu_func(MM, NN, 0.0, 1.0)
    mu_array4 = mu_func(MM, NN, 0.0, -1.0)

    def mu_mat(m: int, n: int, mu_array):
        x = mu_array.shape[0]//2+m
        y = mu_array.shape[1]//2+n
        if x < 0 or y < 0 or x >= mu_array.shape[0] or y >= mu_array.shape[1]:
            raise ValueError(f'mu_mat {x},{y} out of range')
        return mu_array[x, y]
    mu_mat = np.vectorize(mu_mat, excluded=['mu_array'])

    def nu_func(MM, NN, r, s):
        xi_mat_mr_ns = xi_mat(MM.flatten()-r, NN.flatten()-s)
        nu = eng_m.nu_func(xi_mat_mr_ns, z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, z_lst, eps_lst)
        nu = nu.reshape(MM.shape)
        return np.array(nu, dtype=np.complex128)

    nu_array1 = nu_func(MM, NN, 1, 0)
    nu_array2 = nu_func(MM, NN, -1, 0)
    nu_array3 = nu_func(MM, NN, 0, 1)
    nu_array4 = nu_func(MM, NN, 0, -1)

    def nu_mat(m: int, n: int, nu_array):
        x = nu_array.shape[0]//2+m
        y = nu_array.shape[1]//2+n
        if x < 0 or y < 0 or x >= nu_array.shape[0] or y >= nu_array.shape[1]:
            raise ValueError(f'nu_mat {x},{y} out of range')
        return nu_array[x, y]
    nu_mat = np.vectorize(nu_mat, excluded=['nu_array'])


    def varsigma_func(m, n):
        # coeff_mat = np.array(((n, m), (-m, n)))
        # munu_mat = np.array(((-m*mu_func(m, n, 1, 0), -m*mu_func(m, n, -1, 0), n*mu_func(m, n, 0, 1), n*mu_func(m, n, 0, -1)),
        #                      (n*nu_func(m, n, 1, 0), n*nu_func(m, n, -1, 0), m*nu_func(m, n, 0, 1), m*nu_func(m, n, 0, -1))))
        munu_mat11 = -m*mu_mat(m, n, mu_array=mu_array1)
        munu_mat12 = -m*mu_mat(m, n, mu_array=mu_array2)
        munu_mat13 = n*mu_mat(m, n, mu_array=mu_array3)
        munu_mat14 = n*mu_mat(m, n, mu_array=mu_array4)
        munu_mat21 = n*nu_mat(m, n, nu_array=nu_array1)
        munu_mat22 = n*nu_mat(m, n, nu_array=nu_array2)
        munu_mat23 = m*nu_mat(m, n, nu_array=nu_array3)
        munu_mat24 = m*nu_mat(m, n, nu_array=nu_array4)
        varsigma_mat11 = 1/(m**2+n**2)*(n*munu_mat11 + m*munu_mat21)
        varsigma_mat12 = 1/(m**2+n**2)*(n*munu_mat12 + m*munu_mat22)
        varsigma_mat13 = 1/(m**2+n**2)*(n*munu_mat13 + m*munu_mat23)
        varsigma_mat14 = 1/(m**2+n**2)*(n*munu_mat14 + m*munu_mat24)
        varsigma_mat21 = 1/(m**2+n**2)*(-m*munu_mat11 + n*munu_mat21)
        varsigma_mat22 = 1/(m**2+n**2)*(-m*munu_mat12 + n*munu_mat22)
        varsigma_mat23 = 1/(m**2+n**2)*(-m*munu_mat13 + n*munu_mat23)
        varsigma_mat24 = 1/(m**2+n**2)*(-m*munu_mat14 + n*munu_mat24)
        varsigma_mat = np.array(((varsigma_mat11, varsigma_mat12, varsigma_mat13, varsigma_mat14),
                                (varsigma_mat21, varsigma_mat22, varsigma_mat23, varsigma_mat24)))
        return varsigma_mat


    def chi_func(p, q, r, s, direction, order=order):
        sum_raw = 0
        for m in range(-order, order+1):
            for n in range(-order, order+1):
                if m**2+n**2 > 1:
                    if direction == 'x':
                        varsigma = varsigma_func(m, n)[0]
                    elif direction == 'y':
                        varsigma = varsigma_func(m, n)[1]
                    if r == 1 and s == 0:
                        sum_raw += xi_mat(p-m, q-n)*varsigma[0]
                    elif r == -1 and s == 0:
                        sum_raw += xi_mat(p-m, q-n)*varsigma[1]
                    elif r == 0 and s == 1:
                        sum_raw += xi_mat(p-m, q-n)*varsigma[2]
                    elif r == 0 and s == -1:
                        sum_raw += xi_mat(p-m, q-n)*varsigma[3]
        return -k0**2/(2*beta_0)*sum_raw


    # Construct the matrix
    C_1D = np.array(((0, kappa_func(2, 0), 0, 0),
                    (kappa_func(-2, 0), 0, 0, 0),
                    (0, 0, 0, kappa_func(0, 2)),
                    (0, 0, kappa_func(0, -2), 0)))
    C_RAD = np.array(((zeta_func(1, 0, 1, 0), zeta_func(1, 0, -1, 0), 0, 0),
                        (zeta_func(-1, 0, 1, 0), zeta_func(-1, 0, -1, 0), 0, 0),
                        (0, 0, zeta_func(0, 1, 0, 1), zeta_func(0, 1, 0, -1)),
                        (0, 0, zeta_func(0, -1, 0, 1), zeta_func(0, -1, 0, -1))))
    C_2D = np.array(((chi_func(1, 0, 1, 0, 'y'), chi_func(1, 0, -1, 0, 'y'), chi_func(1, 0, 0, 1, 'y'), chi_func(1, 0, 0, -1, 'y')),
                    (chi_func(-1, 0, 1, 0, 'y'), chi_func(-1, 0, -1, 0, 'y'), chi_func(-1, 0, 0, 1, 'y'), chi_func(-1, 0, 0, -1, 'y')),
                    (chi_func(0, 1, 1, 0, 'x'), chi_func(0, 1, -1, 0, 'x'), chi_func(0, 1, 0, 1, 'x'), chi_func(0, 1, 0, -1, 'x')),
                    (chi_func(0, -1, 1, 0, 'x'), chi_func(0, -1, -1, 0, 'x'), chi_func(0, -1, 0, 1, 'x'), chi_func(0, -1, 0, -1, 'x'))))
    C= C_1D + C_RAD + C_2D

    # calculate eigenvalues
    import scipy.constants as const
    eig = np.array(eng_m.eig(matlab.double(C, is_complex=True)))[:,0]
    betas = beta_0 + eig
    freq = a*betas/(2*np.pi*n_eff)
    resonant_wavelength = a/freq
    alpha = 2*np.imag(betas)
    Q = 2*np.pi/a/(alpha)
    print('Q: ', Q)
    print('resonant wavelength (um): ',resonant_wavelength)
    print('alpha (1/cm): ', alpha*1e4)
    print('freq (c/a): ', freq)
    freq_lst.append(freq)
    alpha_lst.append(alpha)
# %%
import matplotlib.pyplot as plt
order_arrary = np.array(order_lst)
freq_arrary = np.array(freq_lst)[:,2:]
alpha_arrary = np.array(alpha_lst)[:,2:]
plt.plot(order_arrary, freq_arrary, 'b', label='freq')
plt.xlabel('order')
plt.ylabel('freq (c/a)')
plt.legend()
plt.twinx()
plt.plot(order_arrary, alpha_arrary*1e4, 'r', label='alpha')
plt.ylabel('alpha (1/cm)')
plt.legend(loc=2)
plt.show()


# %%
def isInsideTriangle(x, y, r):
    return eng_m.isInsideTriangle(x, y, r)

MM, NN = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000), indexing='ij')
isInsideTriangle_array = isInsideTriangle(MM, NN, 1.0)
plt.imshow(isInsideTriangle_array)
plt.colorbar()
plt.show()
# %%
