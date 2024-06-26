first_run = True
# %%
import model
import coeff_func
from model.layers import TMM

import numpy as np
import matplotlib.pyplot as plt
# %%

# Define the dielectric constant distribution
a = 298 # nm
eps_bulk = 3.3 # relative dielectric constant
eps_func = model.rect_lattice.eps_circle(0.3, a, a, eps_bulk)
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
# calculate Fourier coefficients in different methods
# step 1: define the calculation class

xi_dft = coeff_func.xi_func_DFT(eps_func, resolution)
xi_trapz = coeff_func.xi_func(eps_func, method='dbltrapezoid', resolution=resolution)
xi_simps = coeff_func.xi_func(eps_func, method='dblsimpson', resolution=resolution)
xi_romb = coeff_func.xi_func(eps_func, method='dblromb', resolution=resolution)
xi_qmc_quad = coeff_func.xi_func(eps_func, method='dblqmc_quad', n_points=resolution)
if first_run: # dblquad is too slow, only run when necessary
    xi_quad = coeff_func.xi_func(eps_func, method='dblquad', epsabs=1e-15, epsrel=1e-6)
# %%
# step 2: visualize the Fourier coefficients
# Timeit
# %%
%timeit -n 1 -r 1 print('DFT xi 1,1 :',xi_dft[fourier_order])
%timeit -n 1 -r 1 print('Trapezoid integral xi 1,1 :',xi_trapz[fourier_order])
%timeit -n 1 -r 1 print('Simpson integral xi 1,1 :',xi_simps[fourier_order])
%timeit -n 1 -r 1 print('Romberg integral xi 1,1 :',xi_romb[fourier_order])
%timeit -n 1 -r 1 print('QMC quad integral xi 1,1 :',xi_qmc_quad[fourier_order])

%timeit -n 1 -r 1 print('Quad integral xi 1,1 :',xi_quad[fourier_order])
print('Quad abs error :',xi_quad._abserr)

# The result shows that the DFT method is the fastest (one time get many orders) but the not accurate. 
# The dblquad method is the slowest but the most accurate (because set the error tolerance).
# Given that simpson method may meet the trouble when function is not smooth, we should evaluate the error of the simpson method and trapezoid with the quad method.
# Finally, the problem is the eps func is not continuous, so the quad method is difficult to converge. So, I separate the integral region to make the function continuous.

# %%
# step 3: Transfer Matrix Method
FF_lst = np.linspace(0.05,0.40,11)
# FF_lst = [0.1]
k0_lst = []

fig, ax = plt.subplots()
ax1 = plt.twinx()
for FF in FF_lst:
    beta_0 = 2*np.pi/0.295
    t_list = [1.5,0.0885,0.1180,0.0590,1.5]
    eps_list = [11.0224,12.8603,FF+(1-FF)*12.7449,12.7449,11.0224]
    tmm_cal = TMM(t_list, eps_list, beta_0)
    tmm_cal.find_modes()
    k0 = tmm_cal.k0
    k0_lst.append(k0)

    z_mesh = np.linspace(-1, 1.5 + 0.0885 + 0.1180 + 0.0590 + 1.5 +1, 5000)
    E_field_s, eps_s = tmm_cal(z_mesh)
    ax.plot(z_mesh, E_field_s)
    ax1.plot(z_mesh, eps_s, linestyle='--')
    
ax.set_ylabel('intensity')
ax.set_xlabel('z (um)')
ax.set_title('E intensity')
# ax.set_title(f'E intensity\nk_0 = {k0}, beta = {beta_0}, t11 = {tmm_cal.t_11}')
plt.show()

fig, ax = plt.subplots()
ax1 = plt.twinx()
wave_length_ls = 2*np.pi/np.array(k0_lst)
ax.plot(FF_lst, k0_lst)
ax1.plot(FF_lst,wave_length_ls, linestyle='--')
ax.set_ylabel('k0')
ax1.set_ylabel('wave length (um)')
ax.set_xlabel('FF (%)')
plt.show()
# %%
first_run = False