# %%
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib widget

# define epsilon function
def epsilon_noda(xmesh, ymesh, zmesh, radius):
    XX, YY, ZZ = np.meshgrid(xmesh, ymesh, zmesh)
    out_eps = np.ones_like(XX)*11.0224

    out_eps = np.where((ZZ >= 0) & (ZZ < 0.0885), 12.8603, out_eps)
    out_eps = np.where((ZZ >= 0.0885) & (ZZ < 0.0885 + 0.1180), 12.7449, out_eps)
    out_eps = np.where((ZZ >= 0.0885 + 0.1180) & (ZZ < 0.0885 + 0.1180 + 0.0590), 12.7449, out_eps)
    out_eps = np.where((ZZ >= 0.0885) & (ZZ < 0.0885 + 0.1180) & (XX**2 + YY**2 <= radius**2), 1, out_eps)
    return out_eps

# %%
# model_test
# 生成网格
a = 0.295
FF = 0.18
r = np.sqrt(FF/np.pi)*a
xmesh = np.linspace(-a/2, a/2, 150)
ymesh = np.linspace(-a/2, a/2, 150)
zmesh = np.linspace(-1.5 , 0.2655+1.5, 1500)
XX, YY, ZZ = np.meshgrid(xmesh, ymesh, zmesh)
# 生成介电常数
eps = epsilon_noda(xmesh, ymesh, zmesh, r)

# %%
plt.figure()
eps_projz = np.average(eps, axis=(0,1))
plt.plot(zmesh, eps_projz)
plt.show()

# %%
import numpy as np
from numba import jit
t_s = np.array([1.5, 0.0885, 0.1180, 0.0590, 1.5], dtype=np.complex128)
eps0_s = np.array([11.0224+0j, 12.8603+0j, 12.7449+0j, 12.7449+0j, 11.0224+0j], dtype=np.complex128)
eps_s = eps0_s * np.array([1, 1, 1 - FF, 1, 1]) + np.ones_like(eps0_s)*np.array([0, 0, FF, 0, 0])

class TMM_para_noda:
    def __init__(self, FF=0.18):
        self.t_s = np.array([1.5, 0.0885, 0.1180, 0.0590, 1.5], dtype=np.complex128)
        self.eps0_s = np.array([11.0224+0j, 12.8603+0j, 12.7449+0j, 12.7449+0j, 11.0224+0j], dtype=np.complex128)
        self.eps_s = self.eps0_s * np.array([1, 1, 1 - FF, 1, 1]) + np.ones_like(self.eps0_s)*np.array([0, 0, FF, 0, 0])
# %%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(TMM_para_noda().eps_s)
plt.plot(TMM_para_noda().eps0_s)
plt.show
# %%
@jit(nopython=True)
def TMM_11(beta, k0=2*np.pi/0.94, layer=(t_s,eps_s)):
    gamma_s = np.sqrt(beta**2 - k0**2*layer[1])
    d_s = layer[0]
    TMM_total = np.eye(2, dtype=np.complex128)
    for i in range(len(d_s)-1):
        prime0 = np.exp(gamma_s[i]*d_s[i])
        prime1 = np.exp(-gamma_s[i]*d_s[i])
        t00 = (1+gamma_s[i]/gamma_s[i+1])*prime0/2
        t01 = (1-gamma_s[i]/gamma_s[i+1])*prime1/2
        t10 = (1-gamma_s[i]/gamma_s[i+1])*prime0/2
        t11 = (1+gamma_s[i]/gamma_s[i+1])*prime1/2
        TMM_i_raw = np.array(((t00, t01), 
                              (t10, t11)), dtype=np.complex128)

        # TMM_i_raw = np.array([[(gamma_s[i+1]+gamma_s[i])*prime0/2, (gamma_s[i+1]-gamma_s[i])*prime1/2], 
        #                   [(gamma_s[i+1]-gamma_s[i])*prime0/2, (gamma_s[i+1]+gamma_s[i])*prime1/2]])

        TMM_total = np.dot(TMM_i_raw, TMM_total)


    return TMM_total[0,0]
    
        
# %%
import matplotlib.pyplot as plt
# %matplotlib widget
from matplotlib.colors import LogNorm
plt.figure()
beta_list = []
TMM_11_list = []
beta_mesh = np.linspace(0.2794, 0.2796, 1000)
beta_i_mesh = np.linspace(-0.0001, 0.0001, 100)
for beta in beta_mesh:
    beta_c_list = []
    TMM_11_c_list = []
    for beta_i in beta_i_mesh:
        beta_c = beta+beta_i*1j
        beta_c_list.append(beta_c)
        TMM_11_c_list.append(TMM_11(2*np.pi/beta_c))
    beta_list.append(beta_c_list)
    TMM_11_list.append(TMM_11_c_list)
TMM_11_array = np.array(TMM_11_list)
plt.imshow(np.abs(TMM_11_array), aspect='auto', origin='lower', norm=LogNorm(), extent=[beta_i_mesh[0], beta_i_mesh[-1], beta_mesh[0], beta_mesh[-1]])
plt.colorbar()
plt.show()
# %%
from scipy.optimize import minimize
@jit(nopython=True)
def TMM_11_abs(beta_cl):
    beta_c = 2*np.pi/(beta_cl[0]+beta_cl[1]*1j)
    return np.abs(TMM_11(beta_c))

res = minimize(TMM_11_abs, (2.79515525e-01, -8.938e-19), method='Nelder-Mead',tol=1e-10)
print(res['fun'],res['x'])
beta_sol = 2*np.pi/(res['x'][0]+res['x'][1]*1j)

# %%

@jit(nopython=True)
def TMM_mode(z, beta, k0=2*np.pi/0.94, layer=(t_s,eps_s)):
    gamma_s = np.sqrt(beta**2 - k0**2*layer[1])
    d_s = layer[0]
    t_s = np.cumsum(d_s)
    t_s = np.real(t_s)
    A0 = np.array([[1],[0]], dtype=np.complex128)
    TMM_total = np.eye(2, dtype=np.complex128)
    for num_layer in range(len(d_s)):
        if z <= t_s[num_layer]:
            break
    eps = layer[1][num_layer]
    for i in range(num_layer):
        prime0 = np.exp(gamma_s[i]*d_s[i])
        prime1 = np.exp(-gamma_s[i]*d_s[i])
        t00 = (1+gamma_s[i]/gamma_s[i+1])*prime0/2
        t01 = (1-gamma_s[i]/gamma_s[i+1])*prime1/2
        t10 = (1-gamma_s[i]/gamma_s[i+1])*prime0/2
        t11 = (1+gamma_s[i]/gamma_s[i+1])*prime1/2
        TMM_i_raw = np.array(((t00, t01), 
                              (t10, t11)), dtype=np.complex128)
        TMM_total = np.dot(TMM_i_raw, TMM_total)
    T_vec = np.dot(TMM_total, A0)
    print(t_s[0])
    A_i, B_i = T_vec[0,0], T_vec[1,0]
    if z <= t_s[0]:
        E = A_i*np.exp(gamma_s[num_layer]*(z-0)) + B_i*np.exp(-gamma_s[num_layer]*(z-0))
    else:
        E = A_i*np.exp(gamma_s[num_layer]*(z-t_s[num_layer-1])) + B_i*np.exp(-gamma_s[num_layer]*(z-t_s[num_layer-1]))

    return E, eps

# %%
zmesh = np.linspace(0 , 0.2655+3, 1500)
E_ls = []
for z in zmesh:
    E_ls.append(TMM_mode(z, beta_sol))
E_array = np.array(E_ls)
plt.figure()
plt.plot(zmesh, np.real(E_array[:,0]))
plt.ylabel('E')
plt.xlabel('$z\ (\mu m)$')
plt.title(f'Mode profile, neff={np.round(beta_sol/(2*np.pi/0.94), 4)}')
ax = plt.twinx()
plt.plot(zmesh, np.real(E_array[:,1]), color='r')
plt.ylabel('$\epsilon$')

plt.show()
# %%
import pickle
zmesh = zmesh - 1.5
with open('./data/E_noda.pkl', 'wb') as f:
    pickle.dump((E_array,zmesh), f)
# %%
