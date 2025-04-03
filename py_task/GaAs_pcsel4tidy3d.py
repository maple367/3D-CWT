# %%
from model import AlxGaAs, rect_lattice, model_parameters, Model, CWT_solver, SGM_solver, SEMI_solver, user_defined_material
import numpy as np

def start_solver(cores=8):
    import mph
    client = mph.start(cores=cores)
    semi_solver = SEMI_solver(client)
    sgm_solver = SGM_solver(client)
    return semi_solver, sgm_solver

def run_simu(FF ,solvers:list[SEMI_solver|SGM_solver], shape='CC'):
    # semi_solver = solvers[0]
    sgm_solver = solvers[1]
    # Al_x = [0.0, 0.1, 0.0, 0.4, 0.157, 0.45, 0.0]
    eps_list = [3.4824, 3.4461, 3.4824, 3.4824, 3.2744, 3.4118, 3.3799, 3.2499]
    eps_list = np.square(eps_list)
    t_list = [0.12, 0.23, 0.05, 0.03, 0.025, 0.076, 0.04, 1.11]
    is_phc = [True, True, True, False, False, False, False, False]
    is_no_doping = [False, False, False, False, False, True, False, False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            if shape == 'CC':
                rel_r = np.sqrt(FF/np.pi)
                mat_list.append((rect_lattice.eps_circle(rel_r, user_defined_material(eps_list[i]))))
            elif shape == 'RIT':
                rel_r = np.sqrt(FF*2)
                mat_list.append((rect_lattice.eps_ritriangle(rel_r, user_defined_material(eps_list[i]))))
        else:
            mat_list.append(user_defined_material(eps_list[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[17.7, -3.23, 8.28, 2.00]}
    paras = model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = Model(paras)
    # print(pcsel_model.gamma_phc)
    # pcsel_model.plot()
    cwt_solver = CWT_solver(pcsel_model)
    cwt_solver.run(10, parallel=True)
    res = cwt_solver.save_dict
    if cwt_solver.a > 0.5:
        return [{'FF': FF, 'Q': np.nan, 'SE': np.nan, 'shape': shape, 'uuid': paras.uuid}]
    model_size_list = np.array([700])
    i_eigs_inf = np.argmin(np.real(res['eigen_values']))
    print(res['Q'][i_eigs_inf])
    data_set = []
    for model_size in model_size_list:
        try:
            sgm_solver.run(pcsel_model, res['eigen_values'][i_eigs_inf], model_size, 20)
            Q = np.max(res['beta0'].real/(2*sgm_solver.eigen_values.imag))
            i_eigs = np.argmax(res['beta0'].real/(2*sgm_solver.eigen_values.imag))
            SE = 1-sgm_solver.P_edge/sgm_solver.P_stim
            SE = SE[i_eigs]
        except:
            # bad input parameter, the model is not converge
            Q = np.nan
            SE = np.nan
        data_set.append({'FF': FF, 'Q': Q, 'SE': SE, 'shape': shape, 'uuid': paras.uuid, 'size': model_size})
    print(data_set)
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
    for FF in [0.180864]:
        for shape in ['CC', 'RIT']:
            res = run_simu(FF, solvers, shape=shape)
            datas.append(res)
    print(datas)
