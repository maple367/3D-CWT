# %%
if __name__ == '__main__':
    ### README ###
    ### Don't define any function in this block, otherwise it will be called by the child process and cause error. ###
    import multiprocessing as mp
    mp.freeze_support()
    import model
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    lock = mp.Manager().Lock()

    a = 0.298
    FF_lst = np.linspace(0.08,0.24,17)
    # FF_lst = [0.08]
    dataframe = pd.DataFrame(columns=['FF', 'uuid', 'cal_time'])
    for FF in FF_lst:
        rel_r = np.sqrt(FF/np.pi)
        eps_phc = model.rect_lattice.eps_circle(rel_r, a, a, 12.7449)
        t_list = [1.5,0.0885,0.1180,0.0590,1.5]
        eps_list = [11.0224,12.8603,eps_phc,12.7449,11.0224]
        # eps_list = [11.0224,12.8603,FF+(1-FF)*12.7449,12.7449,11.0224]
        paras = model.model_parameters((t_list, eps_list), lock=lock)
        pcsel_model = model.Model(paras)

        # z_mesh = np.linspace(-1, 1.5 + 0.0885 + 0.1180 + 0.0590 + 1.5 +1, 5000)
        # E_field_s = pcsel_model.e_profile(z=z_mesh)
        # eps_s = pcsel_model.eps_profile(z=z_mesh)
        # x_mesh = np.linspace(0, a, 500)
        # y_mesh = np.linspace(0, a, 500)
        # z_points = np.array([1.5+0.0885+0.1180/2,]) # must be a vector
        # XX, YY = np.meshgrid(x_mesh, y_mesh)
        # eps_mesh_phc = pcsel_model.eps_profile(XX, YY, z_points)[0]

        # color1, color2, fontsize1, fontsize2, fontname = 'mediumblue', 'firebrick', 13, 18, 'serif'
        # fig, ax0 = plt.subplots(figsize=(7,5))
        # fig.subplots_adjust(left=0.12, right=0.86)
        # ax1 = plt.twinx()
        # ax0.plot(z_mesh, E_field_s, color=color1)
        # ax0.tick_params(axis='y', colors=color1, labelsize=10)
        # ax0.fill_between(z_mesh, 0, E_field_s, where=pcsel_model.is_in_phc(z_mesh), alpha=0.4, hatch='//', color='orange')
        # ax1.plot(z_mesh, eps_s, linestyle='--', color=color2)
        # ax1.tick_params(axis='y', colors=color2, labelsize=10)
        # ax0.set_xlabel('z', fontsize=fontsize1, fontname=fontname)
        # ax0.set_ylabel('E profile', fontsize=fontsize1, fontname=fontname, color=color1)
        # ax1.set_ylabel('$\epsilon$', fontsize=fontsize1, fontname=fontname, color=color2)
        # plt.title('', fontsize=fontsize2, fontname=fontname)

        # ax2 = ax0.inset_axes([0.07, 0.14, 0.32, 0.32])
        # im = ax2.imshow(np.real(eps_mesh_phc), cmap='Greys')
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        # cb = fig.colorbar(im, cax=ax2.inset_axes([0, 1.05, 1, 0.2]), orientation='horizontal', label='Epsilon')
        # cb.ax.xaxis.set_ticks_position('top')
        # cb.ax.xaxis.set_label_position('top')
        # plt.show()
        # plt.close()

        cwt_solver = model.CWT_solver(pcsel_model)
        cwt_solver.core_num = 15
        cwt_solver.cal_coupling_martix(10, parallel=True)
        data = {'FF': FF, 'uuid': paras.uuid, 'cal_time': cwt_solver._pre_cal_time}
        dataframe = pd.concat([dataframe, pd.DataFrame(data, index=[1])], ignore_index=True) # index is not important, but must given.
    dataframe.to_csv('FF.csv', index=False)
