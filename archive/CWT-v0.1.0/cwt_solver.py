# %%
# What we need to do:
# 1. Define the layer structure
# 2. Define the location and shape of the air holes
# 3. Define the mode profile
import timeit
# %%
import numpy as np
import numba
from numba import jit
import matplotlib.pyplot as plt
import scipy.integrate as si


# %%
a = 0.295
beta_0 = 2*np.pi/a
FF = 0.16
r = np.sqrt(FF/0.5)*a
lambda0 = 0.94
k0 = 2*np.pi/lambda0

order = 10 # order of the Fourier series expansion

# %%
xmesh = np.linspace(-a/2, a/2, 296)
ymesh = np.linspace(-a/2, a/2, 296)
zmesh = np.linspace(-1.5, 0.0885 + 0.1180 + 0.0590 + 1.5, 3265)
XX, YY, ZZ = np.meshgrid(xmesh, ymesh, zmesh, indexing='xy')

@jit(nopython=True)
def epsilon_noda(XX, YY, ZZ, radius):
    out_eps = np.ones_like(XX)*11.0224
    out_eps = np.where((ZZ >= 0) & (ZZ < 0.0885), 12.8603, out_eps)
    out_eps = np.where((ZZ >= 0.0885) & (ZZ < 0.0885 + 0.1180), 12.7449, out_eps)
    out_eps = np.where((ZZ >= 0.0885 + 0.1180) & (ZZ < 0.0885 + 0.1180 + 0.0590), 12.7449, out_eps)
    out_eps = np.where((ZZ >= 0.0885) & (ZZ < 0.0885 + 0.1180) & (XX**2 + YY**2 <= radius**2), 1, out_eps)
    return out_eps
eps_mat_3d = epsilon_noda(XX, YY, ZZ, r)
z_phc_min = 0.0885
z_phc_max = 0.0885 + 0.1180
# %%
eps_mat_projz = np.average(eps_mat_3d, axis=(0,1))

eps_lst = []
z_lst = []
for _ in range(len(eps_mat_projz)-1):
    if eps_mat_projz[_] != eps_mat_projz[_+1]:
        eps_lst.append(eps_mat_projz[_])
        eps_lst.append(eps_mat_projz[_+1])
        z_lst.append(zmesh[_])
        z_lst.append(zmesh[_+1])
eps_array = np.array(eps_lst)
z_array = np.array(z_lst)

# %%
import pickle
with open('./data/E_noda.pkl', 'rb') as f:
    E_array = pickle.load(f)
# %%
E_profile_raw = E_array[0][:,0]
z_mesh_raw = E_array[1]
intensity_integral = si.trapz(np.abs(E_profile_raw)**2, z_mesh_raw)
E_profile_norm = E_profile_raw/np.sqrt(intensity_integral)

def E_profile(z):
    return np.interp(z, z_mesh_raw, E_profile_norm, left=0, right=0)

gamma_phc = si.quad(lambda z: np.square(np.abs(E_profile(z))), z_phc_min, z_phc_max)[0]
# %%
@jit(nopython=True,cache=True)
def n0_func(z):
    return np.sqrt(np.interp(z, z_array, eps_array))

@jit(nopython=True,cache=True)
def beta_z_func(z):
    return k0*n0_func(z)

@jit(nopython=True,cache=True)
def G_func(z, zp):
    return -1j/(2*beta_z_func(z))*np.exp(-1j*beta_z_func(z)*np.abs(z-zp))

# high order part
@jit(nopython=True,cache=True)
def beta_z_func_h(z, m, n):
    return np.sqrt((m**2 + n**2)*beta_0**2 - k0**2*n0_func(z)**2)

@jit(nopython=True,cache=True)
def G_func_h(z, zp, m, n):
    return -1j/(2*beta_z_func_h(z, m, n))*np.exp(-1j*beta_z_func_h(z, m, n)*np.abs(z-zp))

# high order part end

@jit(nopython=True,cache=True)
def eps_phc_fourier_r(y, x, m, n, eps1=12.7449, eps2=1):
    if x+y <= 0 and x>=-r/2 and y>=-r/2:
        return np.real(eps2*np.exp(-1j*(m*beta_0*x + n*beta_0*y)))
    else:
        return np.real(eps1*np.exp(-1j*(m*beta_0*x + n*beta_0*y)))
eps_phc_fourier_r_vec = np.vectorize(eps_phc_fourier_r, excluded=['m', 'n', 'eps1', 'eps2'])
    
@jit(nopython=True,cache=True)
def eps_phc_fourier_i(y, x, m, n, eps1=12.7449, eps2=1):
    if x+y <= 0 and x>=-r/2 and y>=-r/2:
        return np.imag(eps2*np.exp(-1j*(m*beta_0*x + n*beta_0*y)))
    else:
        return np.imag(eps1*np.exp(-1j*(m*beta_0*x + n*beta_0*y)))

# %%
def xi_func(m, n, eps1=12.7449, eps2=1):
    xi_r = 1/a**2 * si.dblquad(eps_phc_fourier_r, -a/2, a/2, lambda x: -a/2, lambda x: a/2, args=(m, n, eps1, eps2))[0]
    xi_i = 1/a**2 * si.dblquad(eps_phc_fourier_i, -a/2, a/2, lambda x: -a/2, lambda x: a/2, args=(m, n, eps1, eps2))[0]
    return xi_r + 1j*xi_i
xi_func = np.vectorize(xi_func)

m_mesh = np.linspace(-(order+1), (order+1), 2*(order+1)+1, dtype=np.int64)
n_mesh = np.linspace(-(order+1), (order+1), 2*(order+1)+1, dtype=np.int64)
MM, NN = np.meshgrid(m_mesh, n_mesh, indexing='xy')
xi_array = xi_func(MM, NN)

def xi_mat(m: int, n: int, xi_array=xi_array):
    x = xi_array.shape[0]//2+m
    y = xi_array.shape[1]//2+n
    if x < 0 or y < 0 or x >= xi_array.shape[0] or y >= xi_array.shape[1]:
        raise ValueError(f'xi_mat {x},{y} out of range')
    return xi_array[x, y]
xi_mat = np.vectorize(xi_mat, excluded=['xi_array'])

# %%
def kappa_func(m, n):
    kappa = -k0**2/(2*beta_0)*xi_mat(m, n)*gamma_phc
    return kappa

def zeta_func(p, q, r, s):
    zeta_r = -k0**4/(2*beta_0)*si.dblquad(lambda zp, z: np.real(xi_mat(p,q)*xi_mat(-r,-s)*G_func(z, zp)*E_profile(zp)*np.conj(E_profile(z))), z_phc_min, z_phc_max, lambda zp: z_phc_min, lambda zp: z_phc_max)[0]
    zeta_i = -k0**4/(2*beta_0)*si.dblquad(lambda zp, z: np.imag(xi_mat(p,q)*xi_mat(-r,-s)*G_func(z, zp)*E_profile(zp)*np.conj(E_profile(z))), z_phc_min, z_phc_max, lambda zp: z_phc_min, lambda zp: z_phc_max)[0]
    return zeta_r + 1j*zeta_i

# %%
# chi part
m_mesh = np.linspace(-(order), (order), 2*(order)+1, dtype=np.int64)
n_mesh = np.linspace(-(order), (order), 2*(order)+1, dtype=np.int64)
MM, NN = np.meshgrid(m_mesh, n_mesh, indexing='xy')
# %%
def mu_func(m, n, r, s):
    mu_r = k0**2*si.dblquad(lambda zp, z: np.real(xi_mat(m-r,n-s)*G_func_h(z, zp, m, n)*E_profile(zp)*np.conj(E_profile(z))), z_phc_min, z_phc_max, lambda zp: z_phc_min, lambda zp: z_phc_max)[0]
    mu_i = k0**2*si.dblquad(lambda zp, z: np.imag(xi_mat(m-r,n-s)*G_func_h(z, zp, m, n)*E_profile(zp)*np.conj(E_profile(z))), z_phc_min, z_phc_max, lambda zp: z_phc_min, lambda zp: z_phc_max)[0]
    return mu_r + 1j*mu_i
mu_func = np.vectorize(mu_func, excluded=['r', 's'])

mu_array1 = mu_func(MM, NN, 1, 0)
mu_array2 = mu_func(MM, NN, -1, 0)
mu_array3 = mu_func(MM, NN, 0, 1)
mu_array4 = mu_func(MM, NN, 0, -1)
# %%
def mu_mat(m: int, n: int, mu_array):
    x = mu_array.shape[0]//2+m
    y = mu_array.shape[1]//2+n
    if x < 0 or y < 0 or x >= mu_array.shape[0] or y >= mu_array.shape[1]:
        raise ValueError(f'mu_mat {x},{y} out of range')
    return mu_array[x, y]
mu_mat = np.vectorize(mu_mat, excluded=['mu_array'])

# %%
def nu_func(m, n, r, s):
    nu_r = -si.quad(lambda z: np.real(1/n0_func(z)**2*xi_mat(m-r,n-s)*np.square(np.abs(E_profile(z)))), z_phc_min, z_phc_max)[0]
    nu_i = -si.quad(lambda z: np.imag(1/n0_func(z)**2*xi_mat(m-r,n-s)*np.square(np.abs(E_profile(z)))), z_phc_min, z_phc_max)[0]
    return nu_r + 1j*nu_i
nu_func = np.vectorize(nu_func, excluded=['r', 's'])

nu_array1 = nu_func(MM, NN, 1, 0)
nu_array2 = nu_func(MM, NN, -1, 0)
nu_array3 = nu_func(MM, NN, 0, 1)
nu_array4 = nu_func(MM, NN, 0, -1)
# %%
def nu_mat(m: int, n: int, nu_array):
    x = nu_array.shape[0]//2+m
    y = nu_array.shape[1]//2+n
    if x < 0 or y < 0 or x >= nu_array.shape[0] or y >= nu_array.shape[1]:
        raise ValueError(f'nu_mat {x},{y} out of range')
    return nu_array[x, y]
nu_mat = np.vectorize(nu_mat, excluded=['nu_array'])

# %%
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


# %%
# Construct the matrix
C_1D = np.array(((0, kappa_func(2, 0), 0, 0),
                (kappa_func(-2, 0), 0, 0, 0),
                (0, 0, 0, kappa_func(0, 2)),
                (0, 0, kappa_func(0, -2), 0)))
C_rad = np.array(((zeta_func(1, 0, 1, 0), zeta_func(1, 0, -1, 0), 0, 0),
                    (zeta_func(-1, 0, 1, 0), zeta_func(-1, 0, -1, 0), 0, 0),
                    (0, 0, zeta_func(0, 1, 0, 1), zeta_func(0, 1, 0, -1)),
                    (0, 0, zeta_func(0, -1, 0, 1), zeta_func(0, -1, 0, -1))))
C_2D = np.array(((chi_func(1, 0, 1, 0, 'y'), chi_func(1, 0, -1, 0, 'y'), chi_func(1, 0, 0, 1, 'y'), chi_func(1, 0, 0, -1, 'y')),
                (chi_func(-1, 0, 1, 0, 'y'), chi_func(-1, 0, -1, 0, 'y'), chi_func(-1, 0, 0, 1, 'y'), chi_func(-1, 0, 0, -1, 'y')),
                (chi_func(0, 1, 1, 0, 'x'), chi_func(0, 1, -1, 0, 'x'), chi_func(0, 1, 0, 1, 'x'), chi_func(0, 1, 0, -1, 'x')),
                (chi_func(0, -1, 1, 0, 'x'), chi_func(0, -1, -1, 0, 'x'), chi_func(0, -1, 0, 1, 'x'), chi_func(0, -1, 0, -1, 'x'))))
C= C_1D + C_rad + C_2D
# %%
# calculate eigenvalues
Q = np.real(np.linalg.eig(C)[0]+k0)/np.abs(2*np.imag(np.linalg.eig(C)[0]+k0))
print('Q: ', Q)
resonant_wavelength = 2*np.pi/(np.linalg.eig(C)[0]+k0)
print('resonant wavelength (um): ',resonant_wavelength)
# %%
