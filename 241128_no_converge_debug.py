
from model import AlxGaAs, rect_lattice, model_parameters, Model, CWT_solver, SGM_solver, SEMI_solver, user_defined_material
import numpy as np

def start_solver(cores=8):
    import mph
    client = mph.start(cores=cores)
    semi_solver = SEMI_solver(client)
    sgm_solver = SGM_solver(client)
    return semi_solver, sgm_solver

def run_simu(FF ,solvers:list[SEMI_solver|SGM_solver], shape='CC'):
    semi_solver = solvers[0]
    sgm_solver = solvers[1]
    Al_x = [0.0, 0.4, 0.157, 0.45, 0.0]
    t_list = [0.43, 0.025, 0.116, 2.11, 0.5]
    is_phc = [True, False, False, False, False]
    is_no_doping = [False, True, True, True, False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            if shape == 'CC':
                rel_r = np.sqrt(FF/np.pi)
                mat_list.append((rect_lattice.eps_circle(rel_r, AlxGaAs(Al_x[i]))))
            elif shape == 'RIT':
                rel_r = np.sqrt(FF*2)
                mat_list.append((rect_lattice.eps_ritriangle(rel_r, AlxGaAs(Al_x[i]))))
        else:
            mat_list.append(AlxGaAs(Al_x[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[17.7, -3.23, 8.28, 2.00]}
    paras = model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
    print(paras.tmm.conveged, paras.tmm.t_11)
    pcsel_model = Model(paras)
    pcsel_model.plot()
    return {'FF': FF, 'Q': 1, 'SE': 2, 'shape': shape, 'uuid': paras.uuid}
    cwt_solver = CWT_solver(pcsel_model)
    cwt_solver.run(10, parallel=True)
    res = cwt_solver.save_dict
    model_size = int(200/cwt_solver.a) # 200 um
    i_eigs_inf = np.argmin(np.real(res['eigen_values']))
    try:
        sgm_solver.run(pcsel_model, res['eigen_values'][i_eigs_inf], model_size, 20)
        Q = np.max(res['beta0'].real/(2*sgm_solver.eigen_values.imag))
        i_eigs = np.argmax(res['beta0'].real/(2*sgm_solver.eigen_values.imag))
        SE = sgm_solver.P_rad/sgm_solver.P_stim
        SE = SE[i_eigs]
    except:
        # bad input parameter, the model is not converge
        Q = np.nan
        SE = np.nan
    pcsel_model.save()
    data_set = {'FF': FF, 'Q': Q, 'SE': SE, 'shape': shape, 'uuid': paras.uuid}
    return data_set

if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    ### README ###
    ### Don't run any sentence out of this block, otherwise it will be called by the child process and cause error. ###
    import pandas as pd
    import matplotlib.pyplot as plt
    # plt.ion()
    fig, axs = plt.subplots(2,1,figsize=(6,6),sharex='all')
    solvers = start_solver(cores=8)
    datas = []
    for FF in np.linspace(0.05,0.25,21):
        for shape in ['CC', 'RIT']:
            res = run_simu(FF, solvers, shape=shape)
            datas.append(res)
            df = pd.DataFrame(datas)
            for ax in axs: ax.cla()
            for shape_name, group_data in df.groupby('shape'):
                group_data.plot.line(x='FF', y='Q', ax=axs[0], marker='o', label=shape_name)
                group_data.plot.line(x='FF', y='SE', ax=axs[1], marker='o', label=shape_name)
            plt.pause(0.1)
    plt.ioff()
    plt.show()
# %%
