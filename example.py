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
# eps_func = model.rect_lattice.eps_userdefine(lambda x, y: np.sin((x-a/4)*2*np.pi/a)*np.cos((y-a/4)*2*np.pi/a)*0.3+3, a, a)
resolution = 2**11+1
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
    xi_quad = coeff_func.xi_func(eps_func, method='dblquad', epsabs=1e-12, epsrel=1e-6)
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
first_run = False