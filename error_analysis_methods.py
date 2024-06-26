first_run = True
# %%
import model
import coeff_func

import numpy as np
import matplotlib.pyplot as plt
# %%

# Define the dielectric constant distribution
a = 298 # nm
eps_bulk = 3.3 # relative dielectric constant
eps_func = model.rect_lattice.eps_circle(0.3, a, a, eps_bulk)
# eps_func = model.rect_lattice.eps_userdefine(lambda x, y: np.sin(x*2*np.pi/a)*np.cos(y*2*np.pi/a)*0.3+3, a, a)
resolution_0 = 2**8

# %%
# visualize cell
if first_run:
    x_grid = np.linspace(0,a,resolution_0)
    y_grid = np.linspace(0,a,resolution_0)
    XX, YY = np.meshgrid(x_grid,y_grid)
    eps_array = eps_func(XX, YY)
    plt.figure(figsize=(3,3))
    plt.tight_layout()
    plt.imshow(eps_array)
    plt.colorbar()
    plt.show()

# %%
# Calculate Fourier coefficients in different methods
# step 1.1: Define the dblquad method to calculate the Fourier coefficients as the reference
xi_quad = coeff_func.xi_func(eps_func, method='dblquad', epsabs=1e-12, epsrel=1e-8)

# %%
# step 1.2: Define the other methods to calculate the Fourier coefficients in different resolution
resolution_ls = np.round(2**np.linspace(7,12,num=11)).astype(int)+1 # resolution must be 2**n + 1 to match the Romberg method
xi_trapz_ls = []
xi_simps_ls = []
xi_romb_ls = []
for resolution in resolution_ls:
    xi_trapz = coeff_func.xi_func(eps_func, method='dbltrapezoid', resolution=resolution)
    xi_simps = coeff_func.xi_func(eps_func, method='dblsimpson', resolution=resolution)
    xi_romb = coeff_func.xi_func(eps_func, method='dblromb', resolution=resolution)
    xi_trapz_ls.append(xi_trapz)
    xi_simps_ls.append(xi_simps)
    xi_romb_ls.append(xi_romb)
xi_methods_ls = [xi_trapz_ls, xi_simps_ls]
# %%
# step 2: Calculate the Fourier coefficients
fourier_order = (2,0)
xi_quad_val = xi_quad[fourier_order]
xi_methods_vals = np.array([np.array([xi[fourier_order] for xi in xi_methods]) for xi_methods in xi_methods_ls])

xi_methods_vals_r = np.real(xi_methods_vals)
xi_methods_vals_i = np.imag(xi_methods_vals)

xi_methods_err_abs_r = np.abs(xi_methods_vals_r - np.real(xi_quad_val))
xi_methods_err_abs_i = np.abs(xi_methods_vals_i - np.imag(xi_quad_val))

xi_methods_err_rel_r = xi_methods_err_abs_r / np.abs(np.real(xi_quad_val))
xi_methods_err_rel_i = xi_methods_err_abs_i / np.abs(np.imag(xi_quad_val))

labels = ['Trapezoid', 'Simpson']
# %%
# step 3: Visualize the Fourier coefficients
fig, axs = plt.subplots(3,1,figsize=(6,18))
ax = axs[0]
ax.plot(resolution_ls, xi_methods_vals_r.T, label=labels)
ax.axhline(y=np.real(xi_quad_val), color='black', label='Quad')
ax.set_xscale('log')
# ax.set_yscale('symlog')
ax.set_xlabel('Resolution')
ax.set_ylabel('Real part of Fourier coefficient')
ax.legend(loc='upper left')
twinx = ax.twinx()
twinx.plot(resolution_ls, xi_methods_vals_i.T, linestyle='--', label=labels)
twinx.axhline(y=np.imag(xi_quad_val), color='black', linestyle='--', label='Quad')
twinx.set_xscale('log')
# twinx.set_yscale('symlog')
twinx.set_xlim(ax.get_xlim())
# twinx.set_xticks(ax.get_xticks())
# twinx.set_xticklabels(ax.get_xticks())
twinx.set_xlabel('Resolution')
twinx.set_ylabel('Imaginary part of Fourier coefficient')
twinx.legend(loc='upper right')
ax.title.set_text('Absolute value of Fourier coefficient')

ax = axs[1]
ax.plot(resolution_ls, xi_methods_err_abs_r.T, label=labels)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Resolution')
ax.set_ylabel('Real part of Fourier coefficient')
ax.legend(loc='upper left')
twinx = ax.twinx()
twinx.plot(resolution_ls, xi_methods_err_abs_i.T, linestyle='--', label=labels)
twinx.set_xscale('log')
twinx.set_yscale('log')
twinx.set_xlim(ax.get_xlim())
# twinx.set_xticks(ax.get_xticks())
# twinx.set_xticklabels(ax.get_xticks())
twinx.set_xlabel('Resolution')
twinx.set_ylabel('Imaginary part of Fourier coefficient')
twinx.legend(loc='upper right')
ax.title.set_text('Absolute error of Fourier coefficient')

ax = axs[2]
ax.plot(resolution_ls, xi_methods_err_rel_r.T, label=labels)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Resolution')
ax.set_ylabel('Real part of Fourier coefficient')
ax.legend(loc='upper left')
twinx = ax.twinx()
twinx.plot(resolution_ls, xi_methods_err_rel_i.T, linestyle='--', label=labels)
twinx.set_xscale('log')
twinx.set_yscale('log')
twinx.set_xlim(ax.get_xlim())
# twinx.set_xticks(ax.get_xticks())
# twinx.set_xticklabels(ax.get_xticks())
twinx.set_xlabel('Resolution')
twinx.set_ylabel('Imaginary part of Fourier coefficient')
twinx.legend(loc='upper right')
ax.title.set_text('Relative error of Fourier coefficient')

plt.show()

# %%
first_run = False
# %%
