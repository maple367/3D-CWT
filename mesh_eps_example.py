import utils
import numpy as np
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
    cwt_solver.core_num = 20 # Because the limitation of Windows, the core_num should be smaller than 61
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
    data_set = {'Q': Q, 'SE': SE, 'uuid': paras.uuid, 't11': pcsel_model.tmm.t_11, 'time_cost': cwt_solver._pre_cal_time}
    return data_set

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, theta):
    x = x-x0
    y = y-y0
    x_theta = x*np.cos(theta) + y*np.sin(theta)
    y_theta = -x*np.sin(theta) + y*np.cos(theta)
    return np.exp(-(x_theta**2/(2*sigma_x**2) + y_theta**2/(2*sigma_y**2)))
        
def generate_sample_array(x_size, y_size, num_holes, x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s):
    XX, YY = np.meshgrid(np.linspace(0, 1, x_size), np.linspace(0, 1, y_size))
    z = 0.0
    for i in range(num_holes):
        x0 = x0_s[i]
        y0 = y0_s[i]
        sigma_x = sigma_x_s[i]
        sigma_y = sigma_y_s[i]
        theta = theta_s[i]
        z += gaussian_2d(XX, YY, x0, y0, sigma_x, sigma_y, theta)
    return z

if __name__ == '__main__':
    import mph
    client = mph.start(cores=8)
    GaAs_eps = AlxGaAs(0).epsilon
    sgm_solver = SGM_solver(client)
    import csv
    import os
    save_path = 'mesh_data_set_3hole_lessscale.csv'
    header = ['Q', 'SE', 'uuid', 't11', 'time_cost']
    
    if not os.path.exists(save_path):
        with open(save_path, mode='a', newline='',encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    i_iter = 0
    while True:
        num_holes = 3
        x0_s = np.random.rand(num_holes)*0.2+np.array([0.15, 0.15, 0.65])
        y0_s = np.random.rand(num_holes)*0.2+np.array([0.15, 0.65, 0.65])
        sigma_x_s = np.random.rand(num_holes)*0.1+0.05
        sigma_y_s = np.random.rand(num_holes)*0.1+0.05
        theta_s = np.random.rand(num_holes)*2*np.pi
        eps_sample = generate_sample_array(32*10, 32*10, num_holes, x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s)
        FF = np.random.rand()*0.1+0.25
        eps_thresh = np.percentile(eps_sample, (1-FF)*100)
        eps_array = np.where(eps_sample<eps_thresh, GaAs_eps, 1.0)
        eps_array = eps_array.reshape(32,10,32,10)
        eps_array = eps_array.mean(axis=(1,3))
        #import matplotlib.pyplot as plt
        #plt.imshow(eps_array.real)
        #plt.colorbar()
        #plt.show()
        res = run_simu(eps_array, sgm_solver)
        with open(save_path, mode='a', newline='',encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([res[key] for key in header])
        i_iter += 1
        print(f'{i_iter}: {res}')
        if os.path.exists('stop'):
            break
    
