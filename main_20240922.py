import model
import utils
import model.rect_lattice
from model import AlxGaAs
import numpy as np
import matplotlib.pyplot as plt

def run_simu(FF,x0,x1,x2,x5,x6,t1,t2,t3,t5,t6,c1,c2,c3,c4,solvers,plot=False):
    semi_solver = solvers[0]
    sgm_solver = solvers[1]
    rel_r = np.sqrt(FF/np.pi)
    Al_x = [x0, x1, x2, 0.4, 0.157, x5, x6]
    t_list = [0.12, t1, t2, t3, 0.076, t5, t6]
    is_phc = [True, True, False, False, False, False, False]
    is_no_doping = [False, False, True, True, True, True, False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            mat_list.append((model.rect_lattice.eps_circle(rel_r, AlxGaAs(Al_x[i]))))
        else:
            mat_list.append(AlxGaAs(Al_x[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[c1, c2, c3, c4]}
    paras = model.model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = model.Model(paras)
    if plot: plot_model(pcsel_model)
    try:
        semi_solver.run(pcsel_model) # use pcsel model to pass parameter
        PCE_raw, SE_raw = semi_solver.get_result(6)
    except:
        # bad input parameter, the model is not converge
        return 0.0
    cwt_solver = model.CWT_solver(pcsel_model)
    cwt_solver.run(10, parallel=True)
    res = cwt_solver.save_dict
    model_size = int(200/cwt_solver.a) # 200 um
    sgm_solver.run(res, res['eigen_values'][0], model_size, 17)
    i_eigs = np.argmin(np.imag(sgm_solver.eigen_values))
    PCE = (1-sgm_solver.P_edge/sgm_solver.P_stim)/(1+pcsel_model.fc_absorption/(np.imag(sgm_solver.eigen_values[i_eigs])*2))*PCE_raw
    data = {'FF': FF, 'PCE': PCE, 'uuid': paras.uuid, 'cal_time': cwt_solver._pre_cal_time}
    return PCE

def plot_model(input_model:model.Model):
    z_mesh = np.linspace(input_model.tmm.z_boundary[0]-0.5, input_model.tmm.z_boundary[-1]+0.5, 5000)
    E_profile_s = input_model.e_normlized_intensity(z=z_mesh)
    dopings = input_model.doping(z=z_mesh)
    eps_s = input_model.eps_profile(z=z_mesh)
    E_profile_s = E_profile_s / np.max(np.abs(E_profile_s)) * (np.max(np.abs(input_model.paras.avg_epsilons)) - np.min(np.abs(input_model.paras.avg_epsilons))) + np.min(np.abs(input_model.paras.avg_epsilons))
    a_const = input_model.paras.cellsize_x
    x_mesh = np.linspace(0, a_const, 500)
    y_mesh = np.linspace(0, a_const, 500)
    z_points = np.array([(input_model.phc_boundary_l[-1]+input_model.phc_boundary_r[-1])/2,]) # must be a vector
    XX, YY = np.meshgrid(x_mesh, y_mesh)
    eps_mesh_phc = input_model.eps_profile(XX, YY, z_points)[0]

    color1, color2, fontsize1, fontsize2, fontname = 'mediumblue', 'firebrick', 13, 18, 'serif'
    fig, ax0 = plt.subplots(figsize=(7,5))
    fig.subplots_adjust(left=0.12, right=0.86)
    ax1 = plt.twinx()
    ax0.plot(z_mesh, dopings, color=color1)
    ax0.tick_params(axis='y', colors=color1, labelsize=10)
    ax1.plot(z_mesh, eps_s, linestyle='--', color=color2)
    ax1.plot(z_mesh, E_profile_s, linestyle='--')
    ax1.fill_between(z_mesh, np.min(E_profile_s), E_profile_s, where=input_model.is_in_phc(z_mesh), alpha=0.4, hatch='//', color='orange')
    ax1.tick_params(axis='y', colors=color2, labelsize=10)
    ax0.set_xlabel(r'z ($\mu m$)', fontsize=fontsize1, fontname=fontname)
    ax0.set_ylabel(r'Doping ($\mu m^{-3}$)', fontsize=fontsize1, fontname=fontname, color=color1)
    ax0.set_yscale('symlog', linthresh=np.min(dopings[dopings!=0.0]))
    ax1.set_ylabel(r'$\epsilon_r$ and Normalized $|E|^2$', fontsize=fontsize1, fontname=fontname, color=color2)
    plt.title('', fontsize=fontsize2, fontname=fontname)

    ax2 = ax0.inset_axes([0.65, 0.10, 0.24, 0.24])
    im = ax2.imshow(np.real(eps_mesh_phc), cmap='Greys')
    ax2.set_xticks([])
    ax2.set_yticks([])
    cb = fig.colorbar(im, cax=ax2.inset_axes([0, 1.05, 1, 0.2]), orientation='horizontal', label='Epsilon')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    plt.show()
    plt.close()

def start_solver(cores=8):
    import mph
    client = mph.start(cores=cores)
    semi_solver = model.SEMI_solver(client)
    sgm_solver = model.SGM_solver(client)
    return semi_solver, sgm_solver

if __name__ == '__main__':
    ### README ###
    ### Don't run any sentence out of this block, otherwise it will be called by the child process and cause error. ###
    import multiprocessing as mp
    mp.freeze_support()
    solvers = start_solver(cores=8)
    print(run_simu(0.181, 0.0, 0.1, 0.0, 0.2, 0.45, 0.23, 0.08, 0.025, 0.04, 2.110, 17.7, -3.23, 8.28, 2.00, solvers))