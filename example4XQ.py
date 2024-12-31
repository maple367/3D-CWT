import utils
import numpy as np
from model import AlxGaAs, rect_lattice, model_parameters, Model, CWT_solver, SGM_solver
def run_simu(variable, sgm_solver:SGM_solver):
    FF = 0.16 # variable, TODO
    rel_s = np.sqrt(2*FF)
    x0, x1, x2, x5, x6 = 0.0, 0.1, 0.0, 0.2, 0.45 # variable, TODO
    Al_x = [x0, x1, x2, 0.4, 0.157, x5, x6]
    t1, t2, t3, t5, t6 = 0.23, 0.08, 0.025, 0.04, 2.110 # variable, TODO
    t_list = [0.12, t1, t2, t3, 0.076, t5, t6]
    is_phc = [True, True, False, False, False, False, False]
    is_no_doping = [False, False, True, True, True, True, False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            mat_list.append((rect_lattice.eps_ritriangle(rel_s, AlxGaAs(Al_x[i]))))
        else:
            mat_list.append(AlxGaAs(Al_x[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[17.7, -3.23, 8.28, 2.00]}
    paras = model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = Model(paras)
    if pcsel_model.tmm.t_11 >= 1e-4:
        return {'Q': np.nan, 'SE': np.nan, 'uuid': paras.uuid, 't11': pcsel_model.tmm.t_11}
    # pcsel_model.plot()
    cwt_solver = CWT_solver(pcsel_model)
    # cwt_solver.core_num = 80 # Because the limitation of python in Windows, the core_num should be smaller than 61
    cwt_solver.run(10, parallel=True)
    res = cwt_solver.save_dict
    model_size = int(200/cwt_solver.a) # 200 um
    i_eigs_inf = np.argmin(np.real(res['eigen_values']))
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
    pcsel_model.save()
    data_set = {'Q': Q, 'SE': SE, 'uuid': paras.uuid, 't11': pcsel_model.tmm.t_11}
    return data_set

if __name__ == '__main__':
    import pandas as pd
    import mph
    client = mph.start(cores=8)
    GaAs_eps = AlxGaAs(0).epsilon
    sgm_solver = SGM_solver(client)
    i_iter = 0
    data_set = []
    while i_iter < 10000:
        variable = None # Need to be defined
        res = run_simu(variable, sgm_solver)
        data_set.append(res)
        i_iter += 1
        print(f'iter {i_iter}:', res)
        if i_iter%50 == 0:
            df = pd.DataFrame(data_set)
            df.to_csv('example_data_set.csv', index=False)
    
