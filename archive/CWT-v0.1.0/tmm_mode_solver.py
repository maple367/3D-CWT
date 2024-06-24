# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget

# define epsilon function
def epsilon_0(xmesh, ymesh, zmesh, etch_depth, radius):
    XX, YY, ZZ = np.meshgrid(xmesh, ymesh, zmesh)
    out_eps = np.ones_like(XX)

    out_eps = np.where((ZZ >= 1) & (ZZ < 1.1), 3.4824**2, out_eps)
    out_eps = np.where((ZZ >= 1.1) & (ZZ < 1.12), 3.46426**2, out_eps)
    out_eps = np.where((ZZ >= 1.12) & (ZZ < 1.33), 3.4461**2, out_eps)
    out_eps = np.where((ZZ >= 1.33) & (ZZ < 1.35), 3.46425**2, out_eps)
    out_eps = np.where((ZZ >= 1.35) & (ZZ < 1.43), 3.4824**2, out_eps)
    out_eps = np.where((ZZ >= 1.43) & (ZZ < 1.455), 3.2744**2, out_eps)
    out_eps = np.where((ZZ >= 1.455) & (ZZ < 1.475), 3.3799**2, out_eps)
    out_eps = np.where((ZZ >= 1.475) & (ZZ < 1.483), 3.5248**2, out_eps)
    out_eps = np.where((ZZ >= 1.483) & (ZZ < 1.503), 3.3799**2, out_eps)
    out_eps = np.where((ZZ >= 1.503) & (ZZ < 1.511), 3.5248**2, out_eps)
    out_eps = np.where((ZZ >= 1.511) & (ZZ < 1.571), 3.3799**2, out_eps)
    out_eps = np.where((ZZ >= 1.571) & (ZZ < 1.591), 3.3124**2, out_eps)
    out_eps = np.where((ZZ >= 1.591) & (ZZ < 3.641), 3.2449**2, out_eps)
    out_eps = np.where((ZZ >= 3.641) & (ZZ < 3.681), 3.36365**2, out_eps)
    out_eps = np.where(ZZ >= 3.681, 3.4824**2, out_eps)
    # ignore substrate
    # out_eps = np.where(ZZ >= 2.641, 3.4824**2, out_eps)
    out_eps = np.where((ZZ >= 1) & (ZZ < 1 + etch_depth) & (XX**2 + YY**2 <= radius**2), 1**2, out_eps)
    return out_eps


# %%
# model_test
# 生成网格
a = 0.298
roa = 0.24
ethc_depth = 0.4
xmesh = np.linspace(-a/2, a/2, 150)
ymesh = np.linspace(-a/2, a/2, 150)
zmesh = np.linspace(0.5 , 4, 3501)
XX, YY, ZZ = np.meshgrid(xmesh, ymesh, zmesh)
# 生成介电常数
eps = epsilon_0(xmesh, ymesh, zmesh, ethc_depth, a*roa)

# %%
plt.figure()
eps_projz = np.average(eps, axis=(0,1))
plt.plot(zmesh, eps_projz)
plt.show()

# %%
eps_ls = [1, ]
z_ls = [0.5, ]
for _ in range(len(zmesh)-1):
    if eps_projz[_] != eps_projz[_+1]:
        z_ls.append(np.round(zmesh[_+1],3))
        eps_ls.append(eps_projz[_+1])
t_s = np.diff(z_ls)
t_s = np.array(t_s, dtype=np.complex128)
eps_s = np.array(eps_ls[:-1], dtype=np.complex128)


# %%
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

plt.figure()
plt.plot(eps_s)
plt.plot(t_s)
plt.show
# %%
@jit(nopython=True)
def TMM_11(beta, k0=2*np.pi/0.98, layer=(t_s,eps_s)):
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
%matplotlib widget
from matplotlib.colors import LogNorm
plt.figure()
beta_list = []
TMM_11_list = []
beta_mesh = np.linspace(0.298, 0.35, 100000)
# beta_i_mesh = np.linspace(-0.0001, 0.0001, 100)
# for beta in beta_mesh:
#     beta_c_list = []
#     TMM_11_c_list = []
#     for beta_i in beta_i_mesh:
#         beta_c = beta+beta_i*1j
#         beta_c_list.append(beta_c)
#         TMM_11_c_list.append(TMM_11(2*np.pi/beta_c))
#     beta_list.append(beta_c_list)
#     TMM_11_list.append(TMM_11_c_list)
# TMM_11_array = np.array(TMM_11_list)
# plt.imshow(np.abs(TMM_11_array), aspect='auto', origin='lower', norm=LogNorm(), extent=[beta_i_mesh[0], beta_i_mesh[-1], beta_mesh[0], beta_mesh[-1]])
# plt.colorbar()
for beta in beta_mesh:
    TMM_11_list.append(TMM_11(2*np.pi/beta))
    beta_list.append(beta)
plt.plot(beta_list,np.abs(TMM_11_list))
plt.yscale('log')
plt.show()


# %%
from scipy.optimize import minimize
@jit(nopython=True)
def TMM_11_abs(beta_cl):
    beta_c = 2*np.pi/(beta_cl[0]+beta_cl[1]*1j)
    return np.abs(TMM_11(beta_c))

res = minimize(TMM_11_abs, (0.30022, -8.938e-19), method='Nelder-Mead',tol=1e-12)
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
    A_i, B_i = T_vec[0,0], T_vec[1,0]
    if z <= t_s[0]:
        E = A_i*np.exp(gamma_s[num_layer]*(z-0)) + B_i*np.exp(-gamma_s[num_layer]*(z-0))
    else:
        E = A_i*np.exp(gamma_s[num_layer]*(z-t_s[num_layer-1])) + B_i*np.exp(-gamma_s[num_layer]*(z-t_s[num_layer-1]))

    return E, eps

# %%
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
