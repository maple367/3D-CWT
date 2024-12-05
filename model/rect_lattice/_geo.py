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
    eps_func = None
    def __init__(self, rel_r, mat_bulk=AlxGaAs(0.0), mat_hole=Air(), cell_size_x=1.0, cell_size_y=1.0):
        self.rel_r = rel_r
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
        self.mat_bulk = mat_bulk
        self.mat_hole = mat_hole
        self.eps_bulk = mat_bulk.epsilon
        self.eps_hole = mat_hole.epsilon
        self.eps_type = 'circle'
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
    eps_func = None
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
    
class eps_grid(eps_userdefine):
    """
    The rectangular lattice divided into 30x30 smaller rectangular cells.
    Each small cell randomly contains either the bulk material or the hole material.
    
    Parameters
    ----------
    mat_bulk : material_class
        The material for the bulk. Default is AlxGaAs(0.0).
    mat_hole : material_class
        The material for the hole. Default is Air().
    cell_size_x & cell_size_y will be automatically determined by model.TMM.
        (cell_size_x : float
            The size of the cell in the x direction.
        cell_size_y : float
            The size of the cell in the y direction.)
    
    Returns
    -------
    out : class
        Call the class with x and y.
        The class returns the dielectric constant distribution in the cell.
    """
    eps_func = None

    def __init__(self, mat_bulk=AlxGaAs(0.0), mat_hole=Air(), cell_size_x=1.0, cell_size_y=1.0):
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
        self.mat_bulk = mat_bulk
        self.mat_hole = mat_hole
        self.eps_bulk = mat_bulk.epsilon
        self.eps_hole = mat_hole.epsilon
        self.eps_type = 'grid'
        self.num_divisions = 30  # Number of divisions along each axis
        self.build()
    
    def build(self):
        # Determine the size of each small cell in the grid
        self.small_cell_size_x = self.cell_size_x / self.num_divisions
        self.small_cell_size_y = self.cell_size_y / self.num_divisions
        
        # Randomly assign each small cell to either mat_bulk or mat_hole
        self.grid = np.random.choice(
            [self.eps_bulk, self.eps_hole],
            size=(self.num_divisions, self.num_divisions)
        )
        
        # Calculate the average dielectric constant
        FF = np.mean(self.grid == self.eps_hole)  # Fraction of holes
        self.avg_eps = self.eps_bulk * (1 - FF) + self.eps_hole * FF
    
    def eps(self, x, y):
        # Find the indices of the small cell containing the point (x, y)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        ix = ((x % self.cell_size_x) // self.small_cell_size_x).astype(int)
        iy = ((y % self.cell_size_y) // self.small_cell_size_y).astype(int)
        # Return the dielectric constant of that small cell
        return np.array([self.grid[_ix, _iy] for _ix, _iy in zip(ix, iy)])

    def __call__(self, x, y):
        return self.eps(x, y)