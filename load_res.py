# %%
if __name__ == '__main__':
    # %%
    import utils
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    data_set_file = 'mesh_data_set_3hole.csv'
    df = pd.read_csv(data_set_file)
    # df.plot.scatter(x='Q', y='SE', marker='o', s=3)
    # plt.xscale('log')
    # plt.show()
    # %%
    df = df.dropna()
    i = 0
    datas = []
    for line in df.values:
        para = utils.Data(f'./history_res/{line[2]}').load_para()
        res = utils.Data(f'./history_res/{line[2]}').load_model()
        eps_array = para['materials'][0].eps_distribtion_array
        eps_array_s = [eps_array, eps_array.T, np.flip(eps_array,0), np.flip(eps_array.T,0), np.flip(eps_array,1), np.flip(eps_array.T,1), np.flip(eps_array), np.flip(eps_array.T)]
        for pseduo_eps_array in eps_array_s:
            datas.append({'eps_array': pseduo_eps_array, 'Q': line[0], 'SE': line[1]})
        i += 1
        print(i)
    np.savez_compressed(f'{data_set_file.removesuffix('.csv')}.npz', pd.DataFrame(datas).sample(frac=1))
    # %%
    index = 367
    temp_para = utils.Data(f'./history_res/{df.iloc[index,2]}')
    temp_para.load_para()
    temp_eps_array = temp_para.materials[0].eps_distribtion_array
    plt.imshow(np.real(temp_eps_array))
    plt.show()
    print(utils.Data(f'./history_res/{df.iloc[index,2]}').load_model()['res']['cwt_res']['Q'])
# %%
from model import AlxGaAs, rect_lattice, model_parameters, Model, CWT_solver, SGM_solver
def run_simu(eps_array, sgm_solver:SGM_solver):
    Al_x =         [0.0,  0.0,  0.4,   0.191, 0.45]
    t_list =       [0.35, 0.08, 0.025, 0.116, 2.11]
    is_phc =       [True, False,False, False, False]
    is_no_doping = [False,False,False, True,  False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            mat_list.append(rect_lattice.eps_mesh(eps_array))
        else:
            mat_list.append(AlxGaAs(Al_x[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[17.7, -3.23, 8.28, 2.00]}
    paras = model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = Model(paras)
    # pcsel_model.plot()
    cwt_solver = CWT_solver(pcsel_model)
    # cwt_solver.core_num = 80 # Because the limitation of Windows, the core_num should be smaller than 61
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
    data_set = {'Q': Q, 'SE': SE, 'uuid': paras.uuid}
    return data_set

# %%
if __name__ == '__main__':
    import mph
    client = mph.start(cores=8)
    sgm_solver = SGM_solver(client)
    reses = []
    for eps_array in [temp_eps_array, temp_eps_array.T, np.flip(temp_eps_array,0), np.flip(temp_eps_array.T,0), np.flip(temp_eps_array,1), np.flip(temp_eps_array.T,1), np.flip(temp_eps_array), np.flip(temp_eps_array.T)]:
        # plt.imshow(np.real(eps_array))
        # plt.show()
        res = run_simu(temp_eps_array, sgm_solver)
        print(res)
        reses.append(res)
    np.save('symmetry_test.npy', reses)
# %%
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(np.load('symmetry_test.npy', allow_pickle=True).tolist())
    df.plot.scatter(x='Q', y='SE', marker='o')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.show()
# %%
