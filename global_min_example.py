# Description: This is an example of how to use the model module to calculate the band structure of a photonic crystal.
# All unit in solver is base on micrometer (um), second (s).
# # the doping function example
# z=0 must be p dopiong, z=+\inf must be n doping.
# def doping_func(x, i, a, b, c, d):
#     layer_i = find_layer(x)
#     if layer_i < i:
#         return np.exp(a + b * x)
#     elif x >= i:
#         return np.exp(c + d * x)
#     else:
#         return 0

# %%
if __name__ == '__main__':
    ### README ###
    ### Don't define any function in this block, otherwise it will be called by the child process and cause error. ###
    import multiprocessing as mp
    mp.freeze_support()
    import model
    from model import AlxGaAs
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    lock = mp.Manager().Lock()

    a = 0.298
    # FF_lst = np.linspace(0.08,0.24,17)
    FF_lst = [0.08]
    dataframe = pd.DataFrame(columns=['FF', 'uuid', 'cal_time'])
    for FF in FF_lst:
        rel_r = np.sqrt(FF/np.pi)
        x0, x1, x2, x5, x6 = 0.0, 0.1, 0.0, 0.2, 0.45
        Al_x = [x0, x1, x2, 0.4, 0.157, x5, x6]
        t1, t2, t3, t5, t6 = 0.23, 0.08, 0.025, 0.04, 2.110
        t_list = [0.12, t1, t2, t3, 0.076, t5, t6]
        is_phc = [True, True, False, False, False, False, False]
        is_no_doping = [False, False, True, True, True, False, False]
        eps_list = []
        for i in range(len(is_phc)):
            if is_phc[i]:
                eps_list.append((model.rect_lattice.eps_circle(rel_r, a, a, AlxGaAs(Al_x[i]).epsilon)))
            else:
                eps_list.append(AlxGaAs(Al_x[i]).epsilon)
        doping_para = {'is_no_doping':is_no_doping,'coeff':[17.7, -3.23, 8.28, 2.00]}
        paras = model.model_parameters((t_list, eps_list, doping_para), lock=lock) # input tuple (t_list, eps_list, index where is the active layer)
        pcsel_model = model.Model(paras)
        tmm = pcsel_model.tmm
        def t_11(beta):
            tmm._construct_matrix(beta)
            return tmm.t_11
        
        betas = np.linspace(2*np.pi/0.98*2.5, 2*np.pi/0.98*3.5, 10000)
        t_11_lst = [t_11(beta) for beta in betas]
        plt.plot(betas, np.abs(t_11_lst))
        plt.yscale('log')
        plt.show()