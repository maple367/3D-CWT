import model
import utils
import model.rect_lattice
from model import AlxGaAs
import numpy as np

def run_simu(eps_array, sgm_solver):
    Al_x =         [0.0,  0.0,  0.4,   0.191, 0.45]
    t_list =       [0.35, 0.08, 0.025, 0.116, 2.11]
    is_phc =       [True, False,False, False, False]
    is_no_doping = [False,False,False, True,  False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            mat_list.append(model.rect_lattice.eps_mesh(eps_array))
        else:
            mat_list.append(AlxGaAs(Al_x[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[17.7, -3.23, 8.28, 2.00]}
    paras = model.model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = model.Model(paras)
    # pcsel_model.plot()
    cwt_solver = model.CWT_solver(pcsel_model)
    cwt_solver.run(10, parallel=True)
    res = cwt_solver.save_dict
    model_size = int(200/cwt_solver.a) # 200 um
    i_eigs_inf = np.argmin(np.imag(res['eigen_values']))
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
    data = [eig_real, eig_imag, eig_real_inf, eig_imag_inf]
    return data

if __name__ == '__main__':
    import mph
    client = mph.start(cores=8)
    GaAs_eps = AlxGaAs(0).epsilon
    sgm_solver = model.SGM_solver(client)
    eps_sample = np.random.random_sample((32, 32))
    FF = 0.2
    eps_thresh = np.percentile(eps_sample, FF*100)
    eps_array = np.where(eps_sample>eps_thresh, GaAs_eps, 1.0)
    res = run_simu(eps_array, sgm_solver)
