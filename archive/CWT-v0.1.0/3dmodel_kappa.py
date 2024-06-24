# %%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import scipy.integrate as si
%matplotlib widget


# %%
# Define constants
k0 = 2*np.pi/0.98
a = 0.298
beta_0 = 2*np.pi/a
gamma_phc = 0.05

@jit(nopython=True,cache=True)
def eps_phc_fourier_r(y, x, m, n, eps1=12.7449, eps2=1, r=0.24*a):
    if x**2 + y**2 <= r**2:
        return np.real(eps2*np.exp(-1j*(m*beta_0*x + n*beta_0*y)))
    else:
        return np.real(eps1*np.exp(-1j*(m*beta_0*x + n*beta_0*y)))

@jit(nopython=True,cache=True)
def eps_phc_fourier_i(y, x, m, n, eps1=12.7449, eps2=1, r=0.24*a):
    if x**2 + y**2 <= r**2:
        return np.imag(eps2*np.exp(-1j*(m*beta_0*x + n*beta_0*y)))
    else:
        return np.imag(eps1*np.exp(-1j*(m*beta_0*x + n*beta_0*y)))
    
def xi_func(m, n, eps1=12.7449, eps2=1, r=0.24*a):
    xi_r = 1/a**2 * si.dblquad(eps_phc_fourier_r, -a/2, a/2, lambda x: -a/2, lambda x: a/2, args=(m, n, eps1, eps2, r))[0]
    xi_i = 1/a**2 * si.dblquad(eps_phc_fourier_i, -a/2, a/2, lambda x: -a/2, lambda x: a/2, args=(m, n, eps1, eps2, r))[0]
    return xi_r + 1j*xi_i

def kappa_func(m, n, r=0.24*a):
    kappa = -k0**2/(2*beta_0)*xi_func(m, n, r=r)*gamma_phc
    return kappa

# %%
kappa1_list = []
kappa3_list = []
r_list = np.linspace(0,0.5,50)
for r in r_list*a:
    kappa1 = kappa_func(1, 0, r=r)
    kappa3 = kappa_func(2, 0, r=r)
    kappa1_list.append(kappa1)
    kappa3_list.append(kappa3)

# %%
plt.figure()
plt.plot(r_list, kappa1_list, label='$\\kappa_1$,$\\kappa_{2D}$')
plt.plot(r_list, kappa3_list, label='$\\kappa_3$,$\\kappa_{1D}$')
plt.xlabel('r/a')
plt.ylabel('$\\kappa$')
plt.legend()
plt.twinx()
plt.plot(r_list, np.array(kappa1_list)/np.abs(np.array(kappa3_list)), label='$\\kappa_1/\\kappa_3$', color='C2')
plt.yscale('log')
plt.legend()
plt.show()

# %%
