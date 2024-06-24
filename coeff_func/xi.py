import numpy as np

def xi_func(eps_func, cell_size_x, cell_size_y, resolution):
    T = 1
    len_x = resolution*T
    len_y = resolution*T
    x_mesh = np.linspace(0, cell_size_x*T, len_x)
    y_mesh = np.linspace(0, cell_size_y*T, len_y)
    X, Y = np.meshgrid(x_mesh, y_mesh)
    eps_array = eps_func(X, Y)
    xi_array = np.fft.fft2(eps_array)/(len_x*len_y)
    return xi_array