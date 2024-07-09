first_run = True
# %%
import model
import calculator

import numpy as np
import matplotlib.pyplot as plt
# %%

# Define the dielectric constant distribution
a = 298 # nm
eps_bulk = 3.3 # relative dielectric constant
eps_func = model.rect_lattice.eps_circle(0.238, a, a, eps_bulk)
# eps_func = model.rect_lattice.eps_userdefine(lambda x, y: np.sin((x-a/4)*2*np.pi/a)*np.cos((y-a/4)*2*np.pi/a)*0.3+3, a, a)
resolution = 2**8+1
fourier_order = (1,1)

# %%
# visualize cell
if first_run:
    x_grid = np.linspace(0,a,resolution)
    y_grid = np.linspace(0,a,resolution)
    XX, YY = np.meshgrid(x_grid,y_grid)
    eps_array = eps_func(XX, YY)
    plt.figure(figsize=(3,3))
    plt.tight_layout()
    plt.imshow(eps_array)
    plt.colorbar()
    plt.show()

# %%
# calculate the error of quad method

xi_quad_ls = []
eps_rel_ls = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
for epsrel in eps_rel_ls:
    xi_quad = calculator.xi_calculator(eps_func, method='dblquad', epsabs=1e-15, epsrel=epsrel)
    xi_quad_ls.append(xi_quad)

# %%
# run calculation
import time
eps_val_ls = []
for xi_quad in xi_quad_ls:
    time_start = time.perf_counter()
    xi_quad[fourier_order]
    time_end = time.perf_counter()
    print(f'epsrel={xi_quad.kwargs["epsrel"]}, time={time_end-time_start:.3f}s')
    eps_val_ls.append(xi_quad[fourier_order])
eps_val_ls = np.array(eps_val_ls)
eps_val_ref = eps_val_ls[-1] + 1e-20j # avoid zero division
eps_val_abserr_ls = [xi_quad._abserr for xi_quad in xi_quad_ls]

# %%
# visualize the error
eps_val_r = np.array(eps_val_ls).real
eps_val_i = np.array(eps_val_ls).imag
eps_val_abserr_r = np.abs(np.real(eps_val_ls-eps_val_ref))
eps_val_abserr_i = np.abs(np.imag(eps_val_ls-eps_val_ref)) 
eps_val_relerr_r = eps_val_abserr_r/np.abs(np.real(eps_val_ref))
eps_val_relerr_i = eps_val_abserr_i/np.abs(np.imag(eps_val_ref))

fig, axs = plt.subplots(3,1,figsize=(8,6*3))
ax = axs[0]
ax.plot(eps_rel_ls, eps_val_r, label='real')
ax.set_xscale('log')
twinx = ax.twinx()
twinx.plot(eps_rel_ls, eps_val_i, label='imag', color='orange')
ax.set_xlabel('epsrel')
ax.set_ylabel('real')
twinx.set_ylabel('imag')
ax.legend(ax.get_lines()+twinx.get_lines(), [line.get_label() for line in ax.get_lines()+twinx.get_lines()])

ax = axs[1]
ax.plot(eps_rel_ls, eps_val_abserr_r, label='abserr real')
ax.set_xscale('log')
twinx = ax.twinx()
twinx.plot(eps_rel_ls, eps_val_abserr_i, label='abserr imag', color='orange')
ax.set_xlabel('epsrel')
ax.set_ylabel('abserr real')
twinx.set_ylabel('abserr imag')
ax.legend(ax.get_lines()+twinx.get_lines(), [line.get_label() for line in ax.get_lines()+twinx.get_lines()])
ax.set_yscale('log')
twinx.set_yscale('log')

ax = axs[2]
ax.plot(eps_rel_ls, eps_val_relerr_r, label='relerr real')
ax.set_xscale('log')
twinx = ax.twinx()
twinx.plot(eps_rel_ls, eps_val_relerr_i, label='relerr imag', color='orange')
ax.set_xlabel('epsrel')
ax.set_ylabel('relerr real')
twinx.set_ylabel('relerr imag')
ax.legend(ax.get_lines()+twinx.get_lines(), [line.get_label() for line in ax.get_lines()+twinx.get_lines()])
ax.set_yscale('log')
twinx.set_yscale('log')

plt.show()

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(eps_rel_ls, np.array(eps_val_abserr_ls).real, label='abserr real')
ax.set_xscale('log')
ax.set_yscale('log')
twinx = ax.twinx()
twinx.plot(eps_rel_ls, np.array(eps_val_abserr_ls).imag, label='abserr imag', color='orange')
ax.set_xlabel('epsrel')
ax.set_ylabel('abserr real')
twinx.set_ylabel('abserr imag')
twinx.set_yscale('log')
ax.legend(ax.get_lines()+twinx.get_lines(), [line.get_label() for line in ax.get_lines()+twinx.get_lines()])
plt.show()

# 1e-6 is a good choice
# %%
first_run = False