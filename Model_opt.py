# Constructed by HH and XQ from NOEL
# updated in 2024.7.30

def PCSEL_1(new_x):
    for i in ['pre_check']:
        def __derive_fac1__(tol_, min_tol, max_tol):
            tol_fac = (np.minimum(tol_, max_tol)-min_tol)/(max_tol-min_tol)
            return 1-(1-np.maximum(tol_fac, 0))**2+2*np.minimum(tol_fac, 0)
        def __derive_fac2__(tol_, min_tol, max_tol):
            return 1-((max_tol-np.minimum(tol_, max_tol))/(max_tol-min_tol))**2
        
        for j in range(11): 
            if new_x[j] < 0: new_x[j] = 0.
        for j in range(1,6,1): 
            if new_x[j] > 0.7: new_x[j] = 0.7
        FF, x0, x1, x2, x5, x6, t1, t2, t3, t5, t6, d1, d2, d3, d4 = new_x
        x3, x4, t0, t4 = 0.4, 0.157, 0.12, 0.076

        score_w1 = __derive_fac2__(
            -np.array([FF,  x0, x1, x2, x5, x6,           t1, t2, t3, t5, t6,      d1, d3, d1+(t0+t1)*d2, d3+t6*d2]), 
            -np.array([1/3, 0.70, 0.70, 0.70, 0.70, 0.70, 3.0, 3.0, 3.0, 3.0, 3.0, 21., 21., 21., 21.]),
            -np.array([0.3, 0.65, 0.65, 0.65, 0.65, 0.65, 2.5, 2.5, 2.5, 2.5, 2.5, 20., 20., 20., 20.]))
        score_w2 = __derive_fac2__(
            np.array([FF,   t1, t2, t3, t5, t6]), 
            np.array([0.09, 0.010, 0.010, 0.010, 0.010, 0.010]), 
            np.array([0.10, 0.015, 0.015, 0.015, 0.015, 0.015]))
        # print(score_w1, score_w2)
        if min(score_w1) < 0 or min(score_w2) < 0: return [0., 0.]
        score_w = np.prod(score_w1)*np.prod(score_w2)

    for i in ['simulation']:
        rel_r = np.sqrt(FF/np.pi)
        Al_x = [x0, x1, x2, x3, x4, x5, x6]
        t_list = [t0, t1, t2, t3, t4, t5, t6]
        is_phc = [True, True, False, False, False, False, False]
        is_no_doping = [False, False, True, True, True, True, False]
        mat_list = []
        for i in range(len(is_phc)):
            if is_phc[i]: mat_list.append((model.rect_lattice.eps_circle(rel_r, AlxGaAs(Al_x[i]))))
            else: mat_list.append(AlxGaAs(Al_x[i]))
        doping_para = {'is_no_doping':is_no_doping,'coeff':[d1, d2, d3, d4]}
        paras = model.model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
        pcsel_model = model.Model(paras)
        
        semi_solver.run(pcsel_model) # use pcsel model to pass parameter
        PCE_raw, SE_raw = semi_solver.get_result()
        cwt_solver = model.CWT_solver(pcsel_model)
        cwt_solver.core_num = 16
        cut_off = 3
        cwt_solver.run(cut_off, parallel=True)
        res = cwt_solver.save_dict
        eigen_infinite = res['eigen_values'][np.imag(res['eigen_values']) > 0]
        eigen_infinite = eigen_infinite[np.where(np.imag(eigen_infinite) == np.min(np.imag(eigen_infinite)))[0]].item() # eigen value with the smallest imaginary part
        sgm_solver = utils.SGM(res, eigen_infinite, 200, 25)
        sgm_res = sgm_solver.run(k=6, show_plot=False)
        eigen_finite = sgm_res[0][np.imag(sgm_res[0]) > 0]
        eigen_finite = eigen_finite[np.where(np.imag(eigen_finite) == np.min(np.imag(eigen_finite)))[0]].flatten().item()
        PCE = np.imag(eigen_infinite)/(np.imag(eigen_finite)+pcsel_model.fc_absorption/2)*PCE_raw

    for i in ['picture']:
        t_c = time.strftime("%Y.%m.%d.%H.%M.%S.", time.localtime())
        import matplotlib.pyplot as plt
        z_mesh = np.linspace(-0.5, np.sum(t_list)+0.5, 5000)
        E_profile_s = pcsel_model.e_normlized_intensity(z=z_mesh)
        dopings = pcsel_model.doping(z=z_mesh)
        eps_s = pcsel_model.eps_profile(z=z_mesh)
        E_profile_s = E_profile_s / np.max(np.abs(E_profile_s)) * (np.max(np.abs(paras.avg_epsilons)) - np.min(np.abs(paras.avg_epsilons))) + np.min(np.abs(paras.avg_epsilons))
        x_mesh = np.linspace(0, paras.cellsize_x, 500)
        y_mesh = np.linspace(0, paras.cellsize_y, 500)
        z_points = np.array([0.3/2,]) # must be a vector
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
        ax1.set_ylabel(r'$\epsilon_r$ and Normalized $|E|^2$', fontsize=fontsize1, fontname=fontname, color=color2)
        plt.title(PCE[6], fontsize=fontsize2, fontname=fontname)

        ax2 = ax0.inset_axes([0.65, 0.10, 0.24, 0.24])
        im = ax2.imshow(np.real(eps_mesh_phc), cmap='Greys')
        ax2.set_xticks([])
        ax2.set_yticks([])
        cb = fig.colorbar(im, cax=ax2.inset_axes([0, 1.05, 1, 0.2]), orientation='horizontal', label='Epsilon')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        plt.savefig(recording_route+'/others/'+t_c+'.png')
        plt.close()
        pass

    return [PCE[6]*score_w, PCE[6]]

if __name__ == '__main__':
    for i in ['import']:
        import multiprocessing as mp
        mp.freeze_support()
        import model
        import utils
        import model.rect_lattice
        from model import AlxGaAs
        import numpy as np
        import time
        import warnings
        warnings.filterwarnings("ignore")
        from rec_bb._rec_bb import data_recording
        from opt._input_provider import explore_core, explore_inter, explore_hull
    
    semi_solver = model.SEMI_solver() # initalization, load comsol model file, will cost about 15s. only need to run once.

    recording_route = r'./recording/'+time.strftime("%Y.%m.%d.%H.%M.%S.", time.localtime())
    rec_bb1 = data_recording(scoring_package=PCSEL_1, x_name='none', y_name='none', recording_route=recording_route)
    rec_bb1.recording_ite = 1

    # FF, x0, x1, x2, x5, x6, t1, t2, t3, t5, t6, d1, d2, d3, d4
    # 0.181, 0.0, 0.1, 0.0, 0.2, 0.45, 0.23, 0.08, 0.025, 0.04, 2.110, 17.7, -3.23, 8.28, 2.00
    # doping: np.exp(a + b * x)
    core_step = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.1, 0.1, 0.1, 0.1]
    core_layer = explore_core(bb=rec_bb1.bb, core_step=core_step, re_check_y='none', _stop_=True, name='C:')
    inter_layer = explore_inter(innerlayer=core_layer, inter_step=[i*6 for i in core_step], _stop_=True, name='I:')
    hull_layer = explore_hull(innerlayer=inter_layer, hull_step=[i*36 for i in core_step], pre_check_x='none', name='H:')

    hull_layer.DR1D1_out(init_x=[0.181, 0.0, 0.1, 0.0, 0.2, 0.45, 0.23, 0.08, 0.025, 0.04, 2.110, 17.7, -3.23, 8.28, 2.00], init_y='none') #initial: (0.367519107953383+0j)