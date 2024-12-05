import model
import utils
import model.rect_lattice
from model import AlxGaAs, user_defined_material
import numpy as np

def run_simu(eps_array, sgm_solver):
    Al_x =         [0.0,  0.0,  0.4,   0.191, 0.45]
    t_list =       [0.35, 0.08, 0.025, 0.116, 2.11]
    is_phc =       [True, False,False, False, False]
    is_no_doping = [False,False,False, True,  False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            mat_list.append(model.rect_lattice.eps_circle(0.2))
        else:
            mat_list.append(AlxGaAs(Al_x[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[17.7, -3.23, 8.28, 2.00]}
    paras = model.model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = model.Model(paras)
    return pcsel_model

pcsel_model = run_simu(np.array([[10, 1.0], [1.0, 10]]), None)
pcsel_model.plot()
