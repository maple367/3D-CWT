from model import TMM
import numpy as np
import matplotlib.pyplot as plt

eps_PVK = 2.235**2
eps_Air = 1.0**2
eps_SiO2 = [1.46**2, 'SiO_{2}']
eps_ZnO = [2.0198**2, 'ZnO']
eps_SiN = [2.0232**2, 'Si_{3}N_{4}']
substrates = [eps_SiO2, eps_ZnO, eps_SiN]

z_mesh = np.linspace(0, 0.2, 1000)

data_set = {'name': [], 'eps_sub': [], 'FF_des': [], 'FF_tps': [], 'gamma': []}
for eps_sub in substrates:
    data_set['name'].append(eps_sub[1])
    data_set['eps_sub'].append(eps_sub[0])
    eps_sub = eps_sub[0]
    FF_des = []
    FF_tps = []
    gammas = []
    for eps_avg in np.linspace(eps_PVK, eps_sub, 26):
        FF_de = (eps_PVK-eps_avg)/(eps_PVK-1)
        FF_tp = (eps_PVK-eps_avg)/(eps_PVK-eps_sub)
        tmm_model = TMM([0.0, 0.2, 0.0], [eps_Air, eps_avg, eps_sub], k0=2*np.pi/0.55)
        tmm_model.find_modes()
        if tmm_model.conveged:
            e_norm = tmm_model.e_normlized_intensity(z_mesh)
            plt.plot(z_mesh, e_norm)
            plt.show()
            gamma = np.trapz(e_norm, z_mesh)
            FF_des.append(FF_de)
            FF_tps.append(FF_tp)
            gammas.append(gamma)
    data_set['FF_des'].append(FF_des)
    data_set['FF_tps'].append(FF_tps)
    data_set['gamma'].append(gammas)

np.save('PVK_waveguide.npy', data_set)

# %%
data_set = {'thickness': [], 'FF_des': [], 'FF_tps': [], 'gamma': []}
for thickness in [0.1, 0.2, 0.3]:
    data_set['thickness'].append(thickness)
    z_mesh = np.linspace(0, thickness, 1000)
    FF_des = []
    FF_tps = []
    gammas = []
    for eps_avg in np.linspace(eps_PVK, eps_SiO2[0], 26):
        FF_de = (eps_PVK-eps_avg)/(eps_PVK-1)
        FF_tp = (eps_PVK-eps_avg)/(eps_PVK-eps_SiO2[0])
        tmm_model = TMM([0.0, thickness, 0.0], [eps_Air, eps_avg, eps_SiO2[0]], k0=2*np.pi/0.55)
        tmm_model.find_modes()
        if tmm_model.conveged:
            e_norm = tmm_model.e_normlized_intensity(z_mesh)
            plt.plot(z_mesh, e_norm)
            plt.show()
            gamma = np.trapz(e_norm, z_mesh)
            FF_des.append(FF_de)
            FF_tps.append(FF_tp)
            gammas.append(gamma)
    data_set['FF_des'].append(FF_des)
    data_set['FF_tps'].append(FF_tps)
    data_set['gamma'].append(gammas)

np.save('PVK_waveguide_thickness.npy', data_set)


# %%
FF = 0.042
eps_PVK = 2.235**2*(1-FF) + eps_SiO2[0]*FF
w = np.linspace(0.195, 0.205, 100)
M = 2*w/0.55*np.sqrt(eps_PVK-eps_SiO2[0]) - 1/np.pi*np.sqrt((eps_SiO2[0]-eps_Air)/(eps_PVK-eps_SiO2[0]))+1
plt.plot(w, M)
plt.hlines(2, w[0], w[-1], 'r')
plt.show()

# %%
