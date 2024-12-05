import cProfile, pstats
from pstats import SortKey
import numpy as np
import numba

@numba.njit(cache=True)
def __eps_mesh0__(x, y, cell_size_x, cell_size_y, eps_distribtion_array_flatten, cell_size_y_eps_distribtion_array_shape0, cell_size_x_eps_distribtion_array_shape1, shape0, shape1):
    return eps_distribtion_array_flatten[(np.floor_divide(np.mod(y, cell_size_y), cell_size_y_eps_distribtion_array_shape0)*shape0+np.floor_divide(np.mod(x, cell_size_x), cell_size_x_eps_distribtion_array_shape1)).astype(np.int_)]

@numba.njit(cache=True)
def __eps_mesh1__(x, y, cell_size_x, cell_size_y, eps_distribtion_array_flatten, cell_size_y_eps_distribtion_array_shape0, cell_size_x_eps_distribtion_array_shape1, shape0, shape1):
    x_mapped = np.mod(x, cell_size_x)
    y_mapped = np.mod(y, cell_size_y)
    x_index = x_mapped//cell_size_x_eps_distribtion_array_shape1
    y_index = y_mapped//cell_size_y_eps_distribtion_array_shape0
    index = int(y_index*shape0+x_index)
    return eps_distribtion_array_flatten[index]
__eps_mesh1__ = np.vectorize(__eps_mesh1__, excluded=['eps_distribtion_array_flatten'])
    
@numba.njit(cache=True)
def __eps_mesh2__(x, y, cell_size_x, cell_size_y, eps_distribtion_array_flatten, cell_size_y_eps_distribtion_array_shape0, cell_size_x_eps_distribtion_array_shape1, shape0, shape1):
    x_mapped = x%cell_size_x
    y_mapped = y%cell_size_y
    x_index = x_mapped//cell_size_x_eps_distribtion_array_shape1
    y_index = y_mapped//cell_size_y_eps_distribtion_array_shape0
    index = int(y_index*shape0+x_index)
    return eps_distribtion_array_flatten[index]
__eps_mesh2__ = np.vectorize(__eps_mesh2__, excluded=['eps_distribtion_array_flatten'])

eps_distribtion_array=np.array([[1,1,0],[0,1,1],[1,0,1]])
eps_distribtion_array_flatten=eps_distribtion_array.flatten()
cell_size_y_eps_distribtion_array_shape0=1/eps_distribtion_array.shape[0]
cell_size_x_eps_distribtion_array_shape1=1/eps_distribtion_array.shape[1]
shape0=eps_distribtion_array.shape[0]
shape1=eps_distribtion_array.shape[1]
x=np.random.random_sample((2,2))
y=np.random.random_sample((2,2))
x_flatten = x.flatten()
y_flatten = y.flatten()
size = (100,100)
size_flatten = size[0]*size[1]

with cProfile.Profile() as pr:
    __eps_mesh0__(x_flatten,y_flatten,1,1,eps_distribtion_array_flatten=eps_distribtion_array_flatten,cell_size_y_eps_distribtion_array_shape0=cell_size_y_eps_distribtion_array_shape0,cell_size_x_eps_distribtion_array_shape1=cell_size_x_eps_distribtion_array_shape1,shape0=shape0,shape1=shape1)
    with open( './__speed_test__/eps_mesh0.txt', 'w' ) as f:
        sortkey = SortKey.TIME
        pstats.Stats( pr, stream=f ).strip_dirs().sort_stats("cumtime").print_stats()

%timeit -n 10000 __eps_mesh0__(np.random.random_sample((size_flatten,)),np.random.random_sample((size_flatten,)),1,1,eps_distribtion_array_flatten=eps_distribtion_array_flatten,cell_size_y_eps_distribtion_array_shape0=cell_size_y_eps_distribtion_array_shape0,cell_size_x_eps_distribtion_array_shape1=cell_size_x_eps_distribtion_array_shape1,shape0=shape0,shape1=shape1)

@numba.njit(cache=True)
def __eps_ritriangle__(x, y, cell_size_x, cell_size_y, half_cell_size_x_s_2, half_cell_size_y_s_2, cell_size_y_cell_size_x, eps_hole, eps_bulk):
    x_mapped = np.mod(x, cell_size_x)
    y_mapped = np.mod(y, cell_size_y)
    return np.where((y_mapped<-cell_size_y_cell_size_x*x_mapped+cell_size_y) * (x_mapped>half_cell_size_x_s_2) * (y_mapped>half_cell_size_y_s_2), eps_hole, eps_bulk)

%timeit -n 10000 __eps_ritriangle__(np.random.random_sample(size),np.random.random_sample(size),1,1,0.5,0.5,1,0,1)

@numba.njit(cache=True)
def __eps_circle__(x, y, cell_size_x, cell_size_y, half_cell_size_x, half_cell_size_y, r__2, eps_hole, eps_bulk):
    x_mapped = np.mod(x, cell_size_x)
    y_mapped = np.mod(y, cell_size_y)
    return np.where((x_mapped - half_cell_size_x)**2 + (y_mapped - half_cell_size_y)**2 < r__2, eps_hole, eps_bulk)

%timeit -n 10000 __eps_circle__(np.random.random_sample(size),np.random.random_sample(size),1,1,0.5,0.5,0.25,0,1)