# %%
import model
import utils
import model.rect_lattice
from model import AlxGaAs
import numpy as np
import matplotlib.pyplot as plt

def start_solver(cores=8):
    import mph
    client = mph.start(cores=cores)
    semi_solver = model.SEMI_solver(client)
    sgm_solver = model.SGM_solver(client)
    return semi_solver, sgm_solver

def run_simu(FF,x0,x1,x2,x5,x6,t1,t2,t3,t5,t6,c1,c2,c3,c4,solvers, shape='CC'):
    semi_solver = solvers[0]
    sgm_solver = solvers[1]
    Al_x = [x0, x1, x2, 0.4, 0.157, x5, x6]
    t_list = [0.12, t1, t2, t3, 0.076, t5, t6]
    is_phc = [True, True, False, False, False, False, False]
    is_no_doping = [False, False, True, True, True, True, False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            if shape == 'CC':
                rel_r = np.sqrt(FF/np.pi)
                mat_list.append((model.rect_lattice.eps_circle(rel_r, AlxGaAs(Al_x[i]))))
            elif shape == 'RIT':
                rel_r = np.sqrt(FF*2)
                mat_list.append((model.rect_lattice.eps_ritriangle(rel_r, AlxGaAs(Al_x[i]))))
        else:
            mat_list.append(AlxGaAs(Al_x[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[c1, c2, c3, c4]}
    paras = model.model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = model.Model(paras)
    cwt_solver = model.CWT_solver(pcsel_model)
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
    data = [FF, eig_real, eig_imag, eig_real_inf, eig_imag_inf, shape]
    return data

if __name__ == '__main__':
    ### README ###
    ### Don't run any sentence out of this block, otherwise it will be called by the child process and cause error. ###
    data_set = {'FF': [], 'eig_real': [], 'eig_imag': [], 'eig_real_inf': [], 'eig_imag_inf': [], 'shape': []}
    import multiprocessing as mp
    import pandas as pd
    mp.freeze_support()
    solvers = start_solver(cores=8)
    for FF in np.linspace(0.05,0.25,11):
        res = run_simu(FF, 0.0, 0.1, 0.0, 0.2, 0.45, 0.23, 0.08, 0.025, 0.04, 2.110, 17.7, -3.23, 8.28, 2.00, solvers, shape='CC')
        for i, key in enumerate(data_set.keys()):
            data_set[key].append(res[i])
        res = run_simu(FF, 0.0, 0.1, 0.0, 0.2, 0.45, 0.23, 0.08, 0.025, 0.04, 2.110, 17.7, -3.23, 8.28, 2.00, solvers, shape='RIT')
        for i, key in enumerate(data_set.keys()):
            data_set[key].append(res[i])
    df = pd.DataFrame(data_set)
    df.to_csv('data_set_new_CWT.csv', index=False)