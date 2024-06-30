# %%
import cProfile, pstats
from pstats import SortKey
# %%
import model
import numpy as np
import matplotlib.pyplot as plt
# %%
a = 0.298
FF_lst = np.linspace(0.05,0.40,11)
FF_lst = [0.05]
k0_lst = []

for FF in FF_lst:
    rel_r = np.sqrt(FF/np.pi)
    eps_phc = model.rect_lattice.eps_circle(rel_r, a, a, 12.7449)
    t_list = [1.5,0.0885,0.1180,0.0590,1.5]
    eps_list = [11.0224,12.8603,eps_phc,12.7449,11.0224]
    # eps_list = [11.0224,12.8603,FF+(1-FF)*12.7449,12.7449,11.0224]
    paras = model.model_parameters((t_list, eps_list))
    pcsel_model = model.Model(paras)
    k0 = pcsel_model.k0
    k0_lst.append(k0)
    z_mesh = np.linspace(-1, 1.5 + 0.0885 + 0.1180 + 0.0590 + 1.5 +1, 5000)
    E_field_s, eps_s = pcsel_model.e_profile(z=z_mesh)
    x_mesh = np.linspace(0, a, 500)
    y_mesh = np.linspace(0, a, 500)
    z_points = np.array([1.5+0.0885+0.1180/2,]) # must be a vector
    XX, YY = np.meshgrid(x_mesh, y_mesh)
    eps_mesh_phc = pcsel_model.eps_profile(XX, YY, z_points)[0]
    fig, ax = plt.subplots()
    ax1 = plt.twinx()
    axi = ax.inset_axes([0.1, 0.1, 0.3, 0.3])
    ax.plot(z_mesh, E_field_s)
    ax.fill_between(z_mesh, 0, E_field_s, where=pcsel_model.is_in_phc(z_mesh), alpha=0.4, hatch='//', color='orange')
    ax1.plot(z_mesh, eps_s, linestyle='--')
    im = axi.imshow(np.real(eps_mesh_phc), cmap='Greys')
    axi.set_xticks([])
    axi.set_yticks([])
    ax.set_xlabel('z')
    ax.set_ylabel('E field')
    ax1.set_ylabel('Epsilon')
    cax = axi.inset_axes([0, 1.05, 1, 0.2])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', label='Epsilon')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    plt.show()
    cwt_solver = model.CWT_solver(pcsel_model)
# %%
def func_model():
    XX, YY = np.meshgrid(x_mesh, y_mesh)
    eps_mesh_phc = pcsel_model.eps_profile(XX, YY, [1.5+0.0885+0.1180/2])
    return eps_mesh_phc

def func_eps_profile():
    XX, YY = np.meshgrid(x_mesh, y_mesh)
    eps_mesh_phc = pcsel_model.paras.epsilons[2](XX, YY)
    return eps_mesh_phc
# %%

with cProfile.Profile() as pr:
    func_eps_profile()
    with open( './__speed_test__/eps_direct.txt', 'w' ) as f:
        sortkey = SortKey.TIME
        pstats.Stats( pr, stream=f ).strip_dirs().sort_stats("cumtime").print_stats()

with cProfile.Profile() as pr:
    func_model()
    with open( './__speed_test__/eps_from_model.txt', 'w' ) as f:
        sortkey = SortKey.TIME
        pstats.Stats( pr, stream=f ).strip_dirs().sort_stats("cumtime").print_stats()

# Optimized speed
# %%
with cProfile.Profile() as pr:
    cwt_solver.zeta_calculator((1,0,1,0))
    with open( './__speed_test__/cal_test.txt', 'w' ) as f:
        sortkey = SortKey.TIME
        pstats.Stats( pr, stream=f ).strip_dirs().sort_stats("cumtime").print_stats()

# %%
def __find_layer(z):
    i = 0
    while z >= pcsel_model.tmm._z_boundary_without_lb[i] and i <= pcsel_model.tmm._len_z_boundary-3:
        i += 1
    return i
find_layer = np.vectorize(__find_layer)

def find_layer_np(z):
    res = np.searchsorted(pcsel_model.tmm._z_boundary_without_lb, z, side='right')
    return np.where(z >= pcsel_model.tmm._z_boundary_without_lb[-1], pcsel_model.tmm._len_z_boundary-2, res)

import random
sample_points = np.array([random.Random().random()*4-0.3 for i in range(5000)])
sample_points = [1.5   , 1.5885, 1.7065, 1.7655, 3.2655]
sample_points = 1.6
(find_layer(sample_points) == find_layer_np(sample_points)).all()
%timeit find_layer(sample_points)
%timeit find_layer_np(sample_points)
# %%
# %%
import pickle, numpy
datapath = r'history\725210b2f15d493ebc28e8a636bdea5a\varsigma_matrix(index=(m,n)).npy'
data = np.load(datapath,allow_pickle=True).item()
# test save and load speed
# %%
%%timeit
with open('test.pkl', 'wb') as f:
    pickle.dump(data, f)
# %%
%%timeit
with open('test.pkl', 'rb') as f:
    data = pickle.load(f)
# %%
%%timeit
numpy.save('test.npy', data)
# %%
%%timeit
data = numpy.load('test.npy', allow_pickle=True).item()
# %%
