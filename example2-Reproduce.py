# %% [markdown]
# https://doi.org/10.14989/doctor.k18283
# parameters I used: page 43
# 
# 
# | Layer    | Thickness ($\mu m$) | Dielectric constant |
# | :----: | :----: | :----: |
# |n-clad(AlGaAs)| 1.5| 11.0224|
# |Active| 0.0885 |12.8603|
# |PC |0.1180 |$\epsilon_{av}$ |
# |GaAs |0.0590 |12.7449|
# |p-clad(AlGaAs) |1.5 |11.0224|
# 
# $$ \epsilon_{av} = FF \cdot \epsilon_a + (1-FF) \cdot \epsilon_b$$
# $FF=0.16$, $D=10$, $\epsilon_a=1.0$, $\epsilon_b=12.7449$, $a=295nm$.
# 

# %% [markdown]
# # Result 1: Coupling coefficients (Fourier coefficients)

# %%
import model
import model.rect_lattice
import numpy as np
import matplotlib.pyplot as plt
from model import user_defined_material

def plot_model(input_model:model.Model):
    z_mesh = np.linspace(input_model.tmm.z_boundary[0]-0.5, input_model.tmm.z_boundary[-1]+0.5, 5000)
    E_profile_s = input_model.e_normlized_amplitude(z=z_mesh)
    dopings = input_model.doping(z=z_mesh)
    eps_s = input_model.eps_profile(z=z_mesh)
    E_profile_s = E_profile_s / np.max(np.abs(E_profile_s)) * (np.max(np.abs(input_model.paras.avg_epsilons)) - np.min(np.abs(input_model.paras.avg_epsilons))) + np.min(np.abs(input_model.paras.avg_epsilons))
    a_const = input_model.paras.cellsize_x
    x_mesh = np.linspace(0, a_const, 500)
    y_mesh = np.linspace(0, a_const, 500)
    z_points = np.array([(input_model.phc_boundary_l[-1]+input_model.phc_boundary_r[-1])/2,]) # must be a vector
    XX, YY = np.meshgrid(x_mesh, y_mesh)
    eps_mesh_phc = input_model.eps_profile(XX, YY, z_points)[0]

    color1, color2, fontsize1, fontsize2, fontname = 'mediumblue', 'firebrick', 13, 18, 'serif'
    fig, ax0 = plt.subplots(figsize=(7,5))
    fig.subplots_adjust(left=0.12, right=0.86)
    ax1 = plt.twinx()
    ax0.plot(z_mesh, dopings, color=color1)
    ax0.tick_params(axis='y', colors=color1, labelsize=10)
    ax1.plot(z_mesh, eps_s, linestyle='--', color=color2)
    ax1.plot(z_mesh, E_profile_s, linestyle='--')
    ax1.fill_between(z_mesh, np.min(E_profile_s), E_profile_s, where=input_model.is_in_phc(z_mesh), alpha=0.4, hatch='//', color='orange')
    ax1.tick_params(axis='y', colors=color2, labelsize=10)
    ax0.set_xlabel(r'z ($\mu m$)', fontsize=fontsize1, fontname=fontname)
    ax0.set_ylabel(r'Doping ($\mu m^{-3}$)', fontsize=fontsize1, fontname=fontname, color=color1)
    ax0.set_yscale('symlog', linthresh=np.min(dopings[dopings!=0.0]))
    ax1.set_ylabel(r'$\epsilon_r$ and Normalized $|E|$', fontsize=fontsize1, fontname=fontname, color=color2)
    ax0.set_title('', fontsize=fontsize2, fontname=fontname)

    ax2 = ax0.inset_axes([0.65, 0.10, 0.24, 0.24])
    im = ax2.imshow(np.real(eps_mesh_phc), cmap='Greys', origin='lower')
    ax2.set_xticks([])
    ax2.set_yticks([])
    cb = fig.colorbar(im, cax=ax2.inset_axes([0, 1.05, 1, 0.2]), orientation='horizontal', label='Epsilon')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    return fig, ax0


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    uuid = '7d2dc35e925d4c8cb553ed2df4228fd0'
    paras = model.model_parameters(load_path=rf'D:\Documents\GitHub\3D-CWT\history_res\{uuid}\input_para.npy') # load
    pcsel_model = model.Model(paras)
    fig, ax = plot_model(pcsel_model)
    fig.show()
    cwt_solver = model.CWT_solver(pcsel_model)

    # %%
    kappa_calculator = cwt_solver.kappa_calculator
    m_mesh = np.arange(-3, 4)
    n_mesh = np.arange(-3, 4)
    MM, NN = np.meshgrid(m_mesh, n_mesh)
    ZZ = np.zeros_like(MM, dtype=np.complex128)
    for i in range(len(m_mesh)):
        for j in range(len(n_mesh)):
            ZZ[i,j] = kappa_calculator((m_mesh[i], n_mesh[j]))
            if m_mesh[i] == 0 and n_mesh[j] == 0:
                ZZ[i,j] = 0.0
    ZZ = ZZ*1e4# /5.98 # cm^-1
    fig, ax = plt.subplots(figsize=(5,6))
    cmap = plt.get_cmap('jet', 49)
    im = ax.pcolormesh(MM, NN, np.abs(ZZ), shading='auto', cmap=cmap, edgecolors='black', linewidth=0.01)
    ax.set_aspect('equal')
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.04)
    cb.ax.tick_params(direction='in', length=3)
    cb.ax.xaxis.set_ticks_position('both')
    cb.set_ticks([np.min(np.abs(ZZ)), np.max(np.abs(ZZ))])
    cb.set_label(r'$\kappa\ (cm^{-1})$')
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            plt.text(m_mesh[i], n_mesh[j], f'{np.abs(ZZ[i, j]):.1f}', ha='center', va='center', color='white', size='small')
    x_mesh = np.linspace(0, 3*pcsel_model.paras.cellsize_x, 500)
    y_mesh = np.linspace(0, 3*pcsel_model.paras.cellsize_x, 500)
    z_points = np.array([(pcsel_model.phc_boundary_l[-1]+pcsel_model.phc_boundary_r[-1])/2,]) # must be a vector
    XX, YY = np.meshgrid(x_mesh, y_mesh)
    eps_mesh_phc = pcsel_model.eps_profile(XX, YY, z_points)[0]
    ax2 = ax.inset_axes([0.80, -0.05, 0.28, 0.28])
    im = ax2.imshow(np.real(eps_mesh_phc), cmap='Greys', origin='lower')
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.show()

    # %% [markdown]
    # The value of $\kappa$ is about 5.98 times of the value in the paper. It is may becuase the coupling coefficient is different from the paper. The coupling coefficient is calculated by the following formula:
    # $$\kappa = -\frac{k_0^2}{2\beta_0} \xi \Gamma_{phc}$$

    # %% [markdown]
    # # Result 2: Wave truncation order

    # %%
    norm_freq_ls = []
    alpha_r_ls = []
    cut_off_ls = np.arange(1, 11)
    for cut_off in cut_off_ls:
        cwt_solver.run(cut_off=cut_off, parallel=False) # parallel=False, because the calculation is already done
        alpha_r_ls.append(cwt_solver.alpha_r)
        norm_freq_ls.append(cwt_solver.norm_freq)

    # %%
    fig, ax = plt.subplots()
    ax.plot(cut_off_ls, [_[:2]*1e4 for _ in alpha_r_ls], marker='o')
    ax.set_xlabel('Wave truncation order, D')
    ax.set_ylabel('Radiation constant, $cm^{-1}$')
    ax.set_xlim([cut_off_ls[0], cut_off_ls[-1]])
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(cut_off_ls, [_[:2] for _ in norm_freq_ls], marker='o')
    ax.set_xlabel('Wave truncation order, D')
    ax.set_ylabel('Normalized frequency, (c/a)')
    ax.set_xlim([cut_off_ls[0], cut_off_ls[-1]])
    fig.show()

    # %% [markdown]
    # # Result 3: High order wave
    # %%
    def cal_high_order_wave(cwt_solver:model.CWT_solver, order, direction:str, num_eig=1):
        m, n = order
        xi_rads = cwt_solver.cal_xi_rads_high_order(order)
        eig_vector = cwt_solver.eigen_vectors[:,num_eig]
        e_plus_coeff = n*xi_rads[0]*eig_vector[0]+n*xi_rads[1]*eig_vector[1]+m*xi_rads[2]*eig_vector[2]+m*xi_rads[3]*eig_vector[3]
        e_minus_coeff = -m*xi_rads[0]*eig_vector[0]-m*xi_rads[1]*eig_vector[1]+n*xi_rads[2]*eig_vector[2]+n*xi_rads[3]*eig_vector[3]
        def _cal_high_order_wave(z):
            e_plus = -1/(cwt_solver.model.eps_profile(z=z))*e_plus_coeff*cwt_solver.model.e_normlized_amplitude(z)
            e_minus = np.square(cwt_solver.k0)*e_minus_coeff*np.sum([cwt_solver.model.integrated_func_1d(lambda _z_prime: cwt_solver.model.Green_func_higher_order(z, _z_prime, order)*cwt_solver.model.e_normlized_amplitude(_z_prime), bd[0], bd[1]) for bd in cwt_solver.model._1d_phc_integral_region_])
            mat_1 = np.array([[n, m], [-m, n]])
            mat_2 = np.array([[e_minus,], [e_plus,]])
            e_vec = 1/(np.square(m)+np.square(n))*np.dot(mat_1, mat_2)
            if direction == 'x':
                return e_vec[0]
            elif direction == 'y':
                return e_vec[1]
            else:
                raise ValueError('direction must be x or y')
        cal_high_order_wave_func = np.vectorize(_cal_high_order_wave)
        return cal_high_order_wave_func
    
    def cal_radiative_wave(cwt_solver:model.CWT_solver, direction:str, num_eig=1):
        xi_rads = cwt_solver.cal_xi_rads_high_order((0,0))
        eig_vector = cwt_solver.eigen_vectors[:,num_eig]
        def _cal_radiative_wave(z):
            if direction == 'x':
                e_coeff = xi_rads[2]*eig_vector[2]+xi_rads[3]*eig_vector[3]
            elif direction == 'y':
                e_coeff = xi_rads[0]*eig_vector[0]+xi_rads[1]*eig_vector[1]
            else:
                raise ValueError('direction must be x or y')
            return np.square(cwt_solver.k0)*e_coeff*np.sum([cwt_solver.model.integrated_func_1d(lambda _z_prime: cwt_solver.model.Green_func_fundamental(z, _z_prime)*cwt_solver.model.e_normlized_amplitude(_z_prime), bd[0], bd[1]) for bd in cwt_solver.model._1d_phc_integral_region_])
        return np.vectorize(_cal_radiative_wave)

    def cal_wave(cwt_solver:model.CWT_solver, order, direction:str, num_eig):
        pcsel_model = cwt_solver.model
        z_mesh = np.linspace(pcsel_model.tmm.z_boundary[0]-0.5, pcsel_model.tmm.z_boundary[-1]+0.5, 5000)
        if order == (0,0):
            wave_calculator = cal_radiative_wave(cwt_solver, direction, num_eig)
        else:
            wave_calculator = cal_high_order_wave(cwt_solver, order, direction, num_eig)
        E_profile_s = wave_calculator(z=z_mesh)
        return z_mesh, E_profile_s
    
    z_mesh, e_profile = cal_wave(cwt_solver, (0,0), 'y', 1)
    E_profile_s = np.abs(e_profile)*pcsel_model.eps_profile(z=z_mesh)
    dopings = pcsel_model.doping(z=z_mesh)
    eps_s = pcsel_model.eps_profile(z=z_mesh)
    E_profile_s = E_profile_s / np.max(np.abs(E_profile_s)) * (np.max(np.abs(pcsel_model.paras.avg_epsilons)) - np.min(np.abs(pcsel_model.paras.avg_epsilons))) + np.min(np.abs(pcsel_model.paras.avg_epsilons))
    a_const = pcsel_model.paras.cellsize_x
    x_mesh = np.linspace(0, a_const, 500)
    y_mesh = np.linspace(0, a_const, 500)
    z_points = np.array([(pcsel_model.phc_boundary_l[-1]+pcsel_model.phc_boundary_r[-1])/2,]) # must be a vector
    XX, YY = np.meshgrid(x_mesh, y_mesh)
    eps_mesh_phc = pcsel_model.eps_profile(XX, YY, z_points)[0]

    color1, color2, fontsize1, fontsize2, fontname = 'mediumblue', 'firebrick', 13, 18, 'serif'
    fig, ax0 = plt.subplots(figsize=(7,5))
    fig.subplots_adjust(left=0.12, right=0.86)
    ax1 = plt.twinx()
    ax0.plot(z_mesh, dopings, color=color1)
    ax0.tick_params(axis='y', colors=color1, labelsize=10)
    ax1.plot(z_mesh, eps_s, linestyle='--', color=color2)
    ax1.plot(z_mesh, E_profile_s, linestyle='--')
    ax1.fill_between(z_mesh, np.min(E_profile_s), E_profile_s, where=pcsel_model.is_in_phc(z_mesh), alpha=0.4, hatch='//', color='orange')
    ax1.tick_params(axis='y', colors=color2, labelsize=10)
    ax0.set_xlabel(r'z ($\mu m$)', fontsize=fontsize1, fontname=fontname)
    ax0.set_ylabel(r'Doping ($\mu m^{-3}$)', fontsize=fontsize1, fontname=fontname, color=color1)
    ax0.set_yscale('symlog', linthresh=np.min(dopings[dopings!=0.0]))
    ax1.set_ylabel(r'$\epsilon_r$ and Normalized $|E|$', fontsize=fontsize1, fontname=fontname, color=color2)
    ax0.set_title('', fontsize=fontsize2, fontname=fontname)

    ax2 = ax0.inset_axes([0.65, 0.10, 0.24, 0.24])
    im = ax2.imshow(np.real(eps_mesh_phc), cmap='Greys', origin='lower')
    ax2.set_xticks([])
    ax2.set_yticks([])
    cb = fig.colorbar(im, cax=ax2.inset_axes([0, 1.05, 1, 0.2]), orientation='horizontal', label='Epsilon')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    fig.show()
# %% [markdown]
# # Result 4: result v.s. FF
# %%
import pandas as pd
import utils
data_ls = pd.read_csv('./FF_Q.csv')
FF_ls = np.unique(data_ls['FF'])[7:-25]
alpha_r_ls = []
norm_freq_ls = []
a_ls = []
for FF in FF_ls:
    data = data_ls[data_ls['FF'] == FF]
    uuid = data['uuid'].values[0]
    res = utils.Data(f'./history_res/{uuid}').load_res()
    alpha_r = np.sort(res['alpha_r'])
    norm_freq = np.sort(res['norm_freq'])
    alpha_r_ls.append(alpha_r)
    norm_freq_ls.append(norm_freq)
    a_ls.append(res['a'])
fig, ax = plt.subplots()
ax.plot(FF_ls, [_[:2]*1e4 for _ in alpha_r_ls], marker='o')
ax.set_xlabel('FF')
ax.set_ylabel('Radiation constant, $cm^{-1}$')
fig.show()

fig, ax = plt.subplots()
ax.plot(FF_ls, [_[:2] for _ in norm_freq_ls], marker='o')
ax.set_xlabel('FF')
ax.set_ylabel('Normalized frequency, (c/a)')
fig.show()

    

# %%
