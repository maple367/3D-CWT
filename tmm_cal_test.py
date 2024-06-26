import numpy as np
import matplotlib.pyplot as plt
from model.layers import TMM
from utils.complexdecimal import ComplexDecimal
from decimal import Decimal

FF_lst = np.linspace(0.05,0.40,21)
FF_lst = [0.005]
k0_lst = []
for FF in FF_lst:
    beta_0 = 2*np.pi/0.295
    t_list = [1.5,0.0885,0.1180,0.0590,1.5]
    eps_list = [11.0224,12.8603,FF+(1-FF)*12.7449,12.7449,11.0224]
    tmm_cal = TMM(t_list, eps_list, beta_0)
    tmm_cal.find_modes()
    k0 = tmm_cal.k0
    k0_lst.append(k0)

    import matplotlib.pyplot as plt

    z_mesh = np.linspace(-1, 1.5 + 0.0885 + 0.1180 + 0.0590 + 1.5 +1, 5000)
    E_field_s, eps_s = tmm_cal(z_mesh)
    E_amp_s, eps_s = tmm_cal.e_amplitude(z_mesh)
    E_field_s = np.array(E_field_s)
    plt.plot(z_mesh, (np.real(E_field_s)/np.max(np.real(E_field_s)))**2, 'b-')
    ax1 = plt.twinx()
    plt.ylabel('eps')
    plt.title(f'E intensity\nk_0 = {k0}, beta = {beta_0}, t11 = {tmm_cal.t_11}')
    plt.plot(z_mesh, eps_s, 'r--')
    plt.show()
