# %%
from model import AlxGaN, rect_lattice, model_parameters, Model, CWT_solver, SGM_solver, SEMI_solver, user_defined_material
import numpy as np

def start_solver(cores=8):
    import mph
    client = mph.start(cores=cores)
    semi_solver = SEMI_solver(client)
    sgm_solver = SGM_solver(client)
    return semi_solver, sgm_solver

def run_simu(FF ,solvers, shape='CC'):
    semi_solver = solvers[0]
    sgm_solver = solvers[1]
    n_list = [2.46, 2.38, 2.47, 2.46]
    t_list = [0.385, 0.005, 0.875, 3.0]
    is_phc = [True, False, False, False]
    is_no_doping = [False, True, True, False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            if shape == 'CC':
                rel_r = np.sqrt(FF/np.pi)
                mat_list.append((rect_lattice.eps_circle(rel_r, user_defined_material(n_list[i]**2))))
            elif shape == 'RIT':
                rel_r = np.sqrt(FF*2)
                mat_list.append((rect_lattice.eps_ritriangle(rel_r, user_defined_material(n_list[i]**2))))
        else:
            mat_list.append(user_defined_material(n_list[i]**2))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[17.7, -3.23, 8.28, 2.00]}
    paras = model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.45) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = Model(paras)
    pcsel_model.plot()
    cwt_solver = CWT_solver(pcsel_model)
    cwt_solver.run(10, parallel=True)
    res = cwt_solver.save_dict
    model_size = int(200/cwt_solver.a) # 200 um
    i_eigs_inf = np.argmin(np.real(res['eigen_values']))
    eig_real_inf = np.real(res['eigen_values'][i_eigs_inf])
    eig_imag_inf = np.imag(res['eigen_values'][i_eigs_inf])
    try:
        sgm_solver.run(res, res['eigen_values'][i_eigs_inf], model_size, 17)
        i_eigs = np.argmin(np.imag(sgm_solver.eigen_values))
        eig_real = np.real(sgm_solver.eigen_values[i_eigs])
        eig_imag = np.imag(sgm_solver.eigen_values[i_eigs])
    except:
        # bad input parameter, the model is not converge
        eig_real = 0.0
        eig_imag = 0.0
    data = [FF, eig_real, eig_imag, eig_real_inf, eig_imag_inf, shape, paras.uuid]
    return data

if __name__ == '__main__':
    ### README ###
    ### Don't run any sentence out of this block, otherwise it will be called by the child process and cause error. ###
    data_set = {'FF': [], 'eig_real': [], 'eig_imag': [], 'eig_real_inf': [], 'eig_imag_inf': [], 'shape': [], 'uuid': []}
    import multiprocessing as mp
    import pandas as pd
    mp.freeze_support()
    solvers = start_solver(cores=8)
    for FF in np.linspace(0.05,0.25,11):
        res = run_simu(FF, solvers, shape='CC')
        for i, key in enumerate(data_set.keys()):
            data_set[key].append(res[i])
        res = run_simu(FF, solvers, shape='RIT')
        for i, key in enumerate(data_set.keys()):
            data_set[key].append(res[i])
    df = pd.DataFrame(data_set)
    df.to_csv('GaN_data_set.csv', index=False)