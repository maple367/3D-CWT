# %%
import model
import numpy as np
import matplotlib.pyplot as plt
from pathos import multiprocessing as mp

if __name__ == '__main__':
    mp.freeze_support()
    a = 0.298
    FF_lst = np.linspace(0.05,0.40,11)
    FF_lst = [0.05]
    k0_lst = []

    for FF in FF_lst:
        rel_r = np.sqrt(FF/np.pi)
        eps_phc = model.rect_lattice.eps_circle(rel_r, a, a, 12.7449)
        t_list = [1.5,0.0885,0.1180,0.0590,1.5]
        eps_list = [11.0224,12.8603,eps_phc,12.7449,11.0224]
        # eps_list = [11.0224,12.8603,FF+(1-FF)*12.7449,12.7449,11.0224]
        paras = model.model_parameters((t_list, eps_list))
        pcsel_model = model.Model(paras)
        k0 = pcsel_model.k0
        # k0_lst.append(k0)
        # z_mesh = np.linspace(-1, 1.5 + 0.0885 + 0.1180 + 0.0590 + 1.5 +1, 5000)
        # E_field_s, eps_s = pcsel_model.e_profile(z=z_mesh)
        # x_mesh = np.linspace(0, a, 500)
        # y_mesh = np.linspace(0, a, 500)
        # z_points = np.array([1.5+0.0885+0.1180/2,]) # must be a vector
        # XX, YY = np.meshgrid(x_mesh, y_mesh)
        # eps_mesh_phc = pcsel_model.eps_profile(XX, YY, z_points)[0]
        # fig, ax = plt.subplots()
        # ax1 = plt.twinx()
        # axi = ax.inset_axes([0.1, 0.1, 0.3, 0.3])
        # ax.plot(z_mesh, E_field_s)
        # ax.fill_between(z_mesh, 0, E_field_s, where=pcsel_model.is_in_phc(z_mesh), alpha=0.4, hatch='//', color='orange')
        # ax1.plot(z_mesh, eps_s, linestyle='--')
        # im = axi.imshow(np.real(eps_mesh_phc), cmap='Greys')
        # axi.set_xticks([])
        # axi.set_yticks([])
        # ax.set_xlabel('z')
        # ax.set_ylabel('E field')
        # ax1.set_ylabel('Epsilon')
        # cax = axi.inset_axes([0, 1.05, 1, 0.2])
        # cb = fig.colorbar(im, cax=cax, orientation='horizontal', label='Epsilon')
        # cb.ax.xaxis.set_ticks_position('top')
        # cb.ax.xaxis.set_label_position('top')
        # plt.show()
        cwt_solver = model.CWT_solver(pcsel_model)
        cwt_solver.cal_coupling_martix(3)