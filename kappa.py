# %%
import numpy as np
from scipy import integrate
from scipy import interpolate

resolution = 2**8 # 1/nm high precision using 2**10, utral high precision using 2**12, coarse precision using 2**8
cell_size_x = 298 # nm
cell_size_y = 298 # nm
# generate an array of a circle

def eps_circle(r, eps_bulk, cell_size_x=cell_size_x, cell_size_y=cell_size_y):
    # r = kwargs.get('r', 100)
    # eps_bulk = kwargs.get('eps_bulk', 3.9)
    # cell_size_x = kwargs.get('cell_size_x', 298)
    # cell_size_y = kwargs.get('cell_size_y', 298)
    r__2 = r**2
    half_cell_size_x = cell_size_x/2
    half_cell_size_y = cell_size_y/2
    def periodically_continued(a, b):
        interval = b - a
        return lambda f: lambda x: f((x - a) % interval + a)
    @periodically_continued(0, cell_size_x)
    def _x(x_):
        return x_
    @periodically_continued(0, cell_size_y)
    def _y(y_):
        return y_
    _x = np.vectorize(_x)
    _y = np.vectorize(_y)
    def eps(x_, y_):
        x_ = _x(x_)
        y_ = _y(y_)
        if (x_ - half_cell_size_x)**2 + (y_ - half_cell_size_y)**2 < r__2:
            return 1
        else:
            return eps_bulk
    # def eps(x_, y_):
    #     eps_ = np.cos(x_*2*np.pi/cell_size_x)*np.cos(y_*2*np.pi/cell_size_y)+np.cos(x_*4*np.pi/cell_size_x)*np.cos(y_*2*np.pi/cell_size_y)
    #     return eps_
    eps = np.vectorize(eps)
    return eps
eps = eps_circle(100, 3.34)
# %%
# draw a circle
import matplotlib.pyplot as plt
x_coords = np.linspace(0, cell_size_x, resolution)
y_coords = np.linspace(0, cell_size_y, resolution)
X, Y = np.meshgrid(x_coords, y_coords)
plt.imshow(eps(X, Y))   
plt.colorbar()
plt.show()
# %%
# calculate xi using the formula of integration
# def xi_func(m, n, eps_func, cell_size_x=cell_size_x, cell_size_y=cell_size_y):
#     xi = 1/(cell_size_x*cell_size_y)*integrate.dblquad(lambda x, y: eps_func(x, y)*np.exp(1j*2*np.pi*(m*x/cell_size_x + n*y/cell_size_y)), 0, cell_size_x, 0, cell_size_y)[0]
#     return xi
# xi = np.vectorize(xi_func, excluded=['eps_func', 'cell_size_x', 'cell_size_y'])
# xi_array = xi(1, 1, eps)
# print(xi_array)

# calculate xi using the DFT
def xi_func(eps_func, cell_size_x=cell_size_x, cell_size_y=cell_size_y, resolution=resolution):
    T = 1
    len_x = resolution*T
    len_y = resolution*T
    x_mesh = np.linspace(0, cell_size_x*T, len_x)
    y_mesh = np.linspace(0, cell_size_y*T, len_y)
    X, Y = np.meshgrid(x_mesh, y_mesh)
    eps_array = eps_func(X, Y)
    xi_array = np.fft.fft2(eps_array)/(len_x*len_y)
    return xi_array
# xi = xi_func(eps)
# print(xi[1,1])

# %%
r_ls = np.linspace(10, 140, 50)
kappa_1d_lst = []
kappa_2d_lst = []
for r in r_ls:
    eps = eps_circle(r, 3.1)
    xi_array = xi_func(eps)
    kappa_array = xi_array*0.3
    kappa_1d_lst.append(xi_array[1, 0])
    kappa_2d_lst.append(xi_array[1, 1])
# %%
kappa_1d_array = np.array(kappa_1d_lst)
kappa_1d_array_real = np.abs(kappa_1d_array.real)
kappa_2d_array = np.array(kappa_2d_lst)
kappa_2d_array_real = np.abs(kappa_2d_array.real)
FF = np.pi*r_ls**2/cell_size_x/cell_size_y
x = FF

fig, ax = plt.subplots()
ax.plot(x_coords, kappa_1d_array_real, label='$\kappa_{1D}$')
ax.plot(x_coords, kappa_2d_array_real, label='$\kappa_{2D}$')
ax.set_xlabel('D/a')
ax.set_ylabel('$\kappa$')
twinx = ax.twinx()
kappa_2d_1d = kappa_2d_array_real/kappa_1d_array_real
twinx.plot(x_coords, kappa_2d_1d, label='$\kappa_{2D}/\kappa_{1D}$', color='r')
twinx.set_ylabel('$\kappa_{2D}/\kappa_{1D}$')
twinx.spines['right'].set_color('red')
plt.legend(ax.get_lines() + twinx.get_lines(), [line.get_label() for line in ax.get_lines()] + [line.get_label() for line in twinx.get_lines()])
plt.show()
# %%
# inverse Fourier transform
z = np.fft.ifft2(xi_array)
plt.imshow(z.real)
plt.colorbar()
plt.show()
# %%
