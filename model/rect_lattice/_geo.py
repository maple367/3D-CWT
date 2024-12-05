import numpy as np
from typing import Callable
import numba
from model import Air, AlxGaAs, material_class

class eps_userdefine():
    """
    The rectangular lattice with userdefine eps_func in one period.
    The cell corners are at (0, 0), (cell_size_x, 0), (0, cell_size_y), and (cell_size_x, cell_size_y).
    Please define the hole function in the eos_func. The hole function should be a function of x and y, and return the dielectric constant of the cell.
    Why use this class?
    The eps_func is periodic in x and y, it is easy to define with this class.
    The geometry information is contained in the class.

    Parameters
    ----------
    eps_func : callable
        The callable of the dielectric constant distribution in one period. Input float 2D coordinate and return dielectric constant.
    cell_size_x : float
        The size of the cell in the x direction.
    cell_size_y : float
        The size of the cell in the y direction.

    Returns
    -------
    out : class
        call the class with x and y.
        The class returns the dielectric constant distribution in the cell.
    """
    eps_type = 'userdefine'
    def __init__(self, eps_func:Callable[[float,float], complex], cell_size_x, cell_size_y):
        self.eps_func = eps_func
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
        x_mesh = np.linspace(0, self.cell_size_x, 2**10+1)
        y_mesh = np.linspace(0, self.cell_size_y, 2**10+1)
        XX, YY = np.meshgrid(x_mesh, y_mesh)
        eps_array = self.eps(XX, YY)
        self.avg_eps = np.mean(eps_array)

    def eps(self, x, y):
        x_mapped = np.mod(x, self.cell_size_x)
        y_mapped = np.mod(y, self.cell_size_y)
        return self.eps_func(x, y)
    
    def __call__(self, x, y):
        return self.eps(x, y)
    
    def __getatt__(self, name: str):
        if name == 'FF':
            print('The FF is not supported in eps_userdefine.')
            return None
        
    def __repr__(self):
        return f"eps_class({self.eps_type}: {self.eps_func}, {self.cell_size_x}, {self.cell_size_y}, id: {id(self)})"
    
    def __len__(self):
        return 1

@numba.njit(cache=True)
def __eps_circle__(x, y, cell_size_x, cell_size_y, half_cell_size_x, half_cell_size_y, r__2, eps_hole, eps_bulk):
    x_mapped = np.mod(x, cell_size_x)
    y_mapped = np.mod(y, cell_size_y)
    return np.where((x_mapped - half_cell_size_x)**2 + (y_mapped - half_cell_size_y)**2 < r__2, eps_hole, eps_bulk)

# air hole class
class eps_circle(eps_userdefine):
    """
    The rectangular lattice with circular holes.
    The center of the circular hole is at the center of the cell.
    The cell corners are at (0, 0), (cell_size_x, 0), (0, cell_size_y), and (cell_size_x, cell_size_y).
    The discontinuity of the dielectric constant has been optimized.

    Parameters
    ----------
    rel_r : float
        The relative radius of the circular hole.
    mat_bulk : material_class
        The material of the bulk. Default is AlxGaAs(0.0).
    mat_hole : material_class
        The material of the hole. Default is Air().
    cell_size_x & cell_size_y is not used in final calculation, it will automatically determine by model.TMM.
        (cell_size_x : float
            The size of the cell in the x direction.
        cell_size_y : float
            The size of the cell in the y direction.)

    Returns
    -------
    out : class
        call the class with x and y.
        The class returns the dielectric constant distribution in the cell.
    """
    eps_func = 'internal'
    def __init__(self, rel_r, mat_bulk=AlxGaAs(0.0), mat_hole=Air(), cell_size_x=1.0, cell_size_y=1.0):
        self.rel_r = rel_r
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
        self.mat_bulk = mat_bulk
        self.mat_hole = mat_hole
        self.eps_bulk = mat_bulk.epsilon
        self.eps_hole = mat_hole.epsilon
        self.eps_type = 'CC'
        self.build()
    
    def build(self):
        self.r = self.rel_r*np.sqrt(self.cell_size_x*self.cell_size_y)
        self.r__2 = self.r**2
        self.half_cell_size_x = self.cell_size_x/2
        self.half_cell_size_y = self.cell_size_y/2
        self.FF = np.pi*self.rel_r**2
        self.avg_eps = self.eps_bulk*(1-self.FF) + self.eps_hole*self.FF

    def eps(self, x, y):
        return __eps_circle__(x, y, self.cell_size_x, self.cell_size_y, self.half_cell_size_x, self.half_cell_size_y, self.r__2, self.eps_hole, self.eps_bulk)
    
    def __call__(self, x, y):
        return self.eps(x, y)

@numba.njit(cache=True)
def __eps_ritriangle__(x, y, cell_size_x, cell_size_y, half_cell_size_x_s_2, half_cell_size_y_s_2, cell_size_y_cell_size_x, eps_hole, eps_bulk):
    x_mapped = np.mod(x, cell_size_x)
    y_mapped = np.mod(y, cell_size_y)
    return np.where((y_mapped<-cell_size_y_cell_size_x*x_mapped+cell_size_y) * (x_mapped>half_cell_size_x_s_2) * (y_mapped>half_cell_size_y_s_2), eps_hole, eps_bulk)

class eps_ritriangle(eps_userdefine):
    """
    The rectangular lattice with right-angled isosceles triangle holes.
    The center of the right triangle's long side is at the center of the cell.
    The cell corners are at (0, 0), (cell_size_x, 0), (0, cell_size_y), and (cell_size_x, cell_size_y).
    The discontinuity of the dielectric constant has been optimized.

    Parameters
    ----------
    rel_s : float
        The relative side length of the right-angled isosceles triangular hole.
    mat_bulk : material_class
        The material of the bulk. Default is AlxGaAs(0.0).
    mat_hole : material_class
        The material of the hole. Default is Air().
    cell_size_x & cell_size_y is not used in final calculation, it will automatically determine by model.TMM.
        (cell_size_x : float
            The size of the cell in the x direction.
        cell_size_y : float
            The size of the cell in the y direction.)

    Returns
    -------
    out : class
        call the class with x and y.
        The class returns the dielectric constant distribution in the cell.
    """
    eps_func = 'internal'
    def __init__(self, rel_s, mat_bulk=AlxGaAs(0.0), mat_hole=Air(), cell_size_x=1.0, cell_size_y=1.0):
        self.rel_s = rel_s
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
        self.mat_bulk = mat_bulk
        self.mat_hole = mat_hole
        self.eps_bulk = mat_bulk.epsilon
        self.eps_hole = mat_hole.epsilon
        self.eps_type = 'RIT'
        self.build()
    
    def build(self):
        self.s = self.rel_s*np.sqrt(self.cell_size_x*self.cell_size_y)
        self.s_2 = self.s/2
        self.half_cell_size_x = self.cell_size_x/2
        self.half_cell_size_y = self.cell_size_y/2
        self.half_cell_size_x_s_2 = self.half_cell_size_x-self.s_2
        self.half_cell_size_y_s_2 = self.half_cell_size_y-self.s_2
        self.cell_size_y_cell_size_x = self.cell_size_y/self.cell_size_x
        self.FF = self.rel_s**2/2
        self.avg_eps = self.eps_bulk*(1-self.FF) + self.eps_hole*self.FF

    def eps(self, x, y):
        return __eps_ritriangle__(x, y, self.cell_size_x, self.cell_size_y, self.half_cell_size_x_s_2, self.half_cell_size_y_s_2, self.cell_size_y_cell_size_x, self.eps_hole, self.eps_bulk)
    
    def __call__(self, x, y):
        return self.eps(x, y)

@numba.njit(cache=True)
def __eps_mesh0__(x, y, cell_size_x, cell_size_y, eps_distribtion_array_flatten, cell_size_y_eps_distribtion_array_shape0, cell_size_x_eps_distribtion_array_shape1, shape0) -> np.ndarray:
    return eps_distribtion_array_flatten[(np.floor_divide(np.mod(y, cell_size_y), cell_size_y_eps_distribtion_array_shape0)*shape0+np.floor_divide(np.mod(x, cell_size_x), cell_size_x_eps_distribtion_array_shape1)).astype(np.int_)]

class eps_mesh(eps_userdefine):
    """
    The rectangular lattice with array eps_distribtion_array in one period.
    The cell corners are at (0, 0), (cell_size_x, 0), (0, cell_size_y), and (cell_size_x, cell_size_y).
    TODO: The discontinuity of the dielectric constant need to be optimized.

    Parameters
    ----------
    eps_distribtion_array : np.ndarray
        The dielectric constant distribution in one period.
    cell_size_x & cell_size_y is not used in final calculation, it will automatically determine by model.TMM.
        (cell_size_x : float
            The size of the cell in the x direction.
        cell_size_y : float
            The size of the cell in the y direction.)

    Returns
    -------
    out : class
        call the class with x and y.
        The class returns the dielectric constant distribution in the cell.
    """
    eps_func = 'internal'
    def __init__(self, eps_distribtion_array:np.ndarray, cell_size_x=1.0, cell_size_y=1.0):
        self.eps_distribtion_array = eps_distribtion_array
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
        self.max_eps = self.eps_distribtion_array.max()
        self.min_eps = self.eps_distribtion_array.min()
        self.avg_eps = self.eps_distribtion_array.mean()
        self.FF = (self.max_eps-self.avg_eps)/(self.max_eps-self.min_eps) # effective fill factor
        self.eps_type = 'MESH'
        self.build()
    
    def build(self):
        self.eps_distribtion_array_flatten = self.eps_distribtion_array.flatten()
        self.cell_size_y_eps_distribtion_array_shape0 = 1/self.eps_distribtion_array.shape[0]
        self.cell_size_x_eps_distribtion_array_shape1 = 1/self.eps_distribtion_array.shape[1]
        self.shape0 = self.eps_distribtion_array.shape[0]
        
    def eps(self, x:np.ndarray, y:np.ndarray):
        x = np.array(x)
        input_shape = x.shape
        x = x.flatten()
        y = np.array(y).flatten()
        eps_out =  __eps_mesh0__(x, y, self.cell_size_x, self.cell_size_y, self.eps_distribtion_array_flatten, self.cell_size_y_eps_distribtion_array_shape0, self.cell_size_x_eps_distribtion_array_shape1, self.shape0)
        return eps_out.reshape(input_shape)
    
    def __call__(self, x, y):
        return self.eps(x, y)