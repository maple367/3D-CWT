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
    semi_solver = solvers[0]
    sgm_solver = solvers[1]
    Al_x = [0.0, 0.1, 0.0, 0.4, 0.157, 0.45, 0.0]
    t_list = [0.1, 0.25, 0.08, 0.025, 0.116, 2.11, 0.5]
    is_phc = [True, True, True, False, False, False, False]
    is_no_doping = [False, False, False, True, True, True, False]
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
    pcsel_model = Model(paras)
    # print(pcsel_model.gamma_phc)
    # pcsel_model.plot()
    cwt_solver = CWT_solver(pcsel_model)
    cwt_solver.run(10, parallel=True)
    res = cwt_solver.save_dict
    if cwt_solver.a > 0.5:
        return {'FF': FF, 'Q': np.nan, 'SE': np.nan, 'shape': shape, 'uuid': paras.uuid}
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
    data_set = {'FF': FF, 'Q': Q, 'SE': SE, 'shape': shape, 'uuid': paras.uuid}
    return data_set

if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    ### README ###
    ### Don't run any sentence out of this block, otherwise it will be called by the child process and cause error. ###
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.ion()
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
    df.to_csv('GaAs_data_set.csv', index=False)
    plt.ioff()
    plt.show()

# %%
if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    ### README ###
    ### Don't run any sentence out of this block, otherwise it will be called by the child process and cause error. ###
    import pandas as pd
    import utils
    import numpy as np
    df = pd.read_csv('GaAs_data_set.csv')
    Q_list = []
    SE_list = []
    FF_list = []
    fliter_shape = 'RIT'
    for uuid in df['uuid']:
        try:
            res = utils.Data(f'./history_res/{uuid}').load_model()['res']
        except:
            continue
        cwt_res = res['cwt_res']
        if df[df['uuid']==uuid]['shape'].values[0] != fliter_shape:
            continue
        try:
            sgm_res = res['sgm_res']
            if sgm_res['a'] >= 0.5:
                continue
            index = np.argmax(cwt_res['beta0'].real/(2*sgm_res['eigen_values'].imag))
            Q = cwt_res['beta0'].real/(2*sgm_res['eigen_values'].imag)
            SE = sgm_res['P_edge']/sgm_res['P_stim']
            if SE[index] < 0:
                continue
            Q_list.append(Q[index])
            SE_list.append(1-SE[index])
            FF_list.append(df[df['uuid']==uuid]['FF'].values[0])
        except:
            continue
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(FF_list, Q_list, marker='o', label='Q', color='r')
    twinx = ax.twinx()
    twinx.plot(FF_list, SE_list, marker='o', label=r'$P_{\mathrm{rad}}/P_{\mathrm{stim}}$', color='b')
    ax.legend(loc='center left')
    twinx.legend(loc='center right')
    ax.set_xlabel('FF')
    ax.set_ylabel('Q')
    twinx.set_ylabel(r'$P_{\mathrm{rad}}/P_{\mathrm{stim}}$')
    ax.set_title(fliter_shape)
    plt.show()
# %%
