import numpy as np
import numba as nb
from typing import Callable
from .._boundary import _periodically_continued

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
        @_periodically_continued(0, cell_size_x)
        def _x(x_):
            return x_
        @_periodically_continued(0, cell_size_y)
        def _y(y_):
            return y_
        self._x = _x
        self._y = _y
        x_mesh = np.linspace(0, self.cell_size_x, 2**10+1)
        y_mesh = np.linspace(0, self.cell_size_y, 2**10+1)
        XX, YY = np.meshgrid(x_mesh, y_mesh)
        eps_array = self.eps(XX, YY)
        self.avg_eps = np.mean(eps_array)

    def eps(self, x, y):
        x = self._x(x)
        y = self._y(y)
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


# air hole class
class eps_circle(eps_userdefine):
    """
    The rectangular lattice with circular holes.
    The center of the circular hole is at the center of the cell.
    The cell corners are at (0, 0), (cell_size_x, 0), (0, cell_size_y), and (cell_size_x, cell_size_y).
    The discontinuity of the dielectric constant has been optimized.
    Parameters
    ----------
    r : float
        The radius of the circular hole.
    cell_size_x : float
        The size of the cell in the x direction.
    cell_size_y : float
        The size of the cell in the y direction.
    eps_bulk : float
        The dielectric constant of the bulk material.
    eps_hole : float
        The dielectric constant of the hole. Default is 1.0 (air).

    Returns
    -------
    out : class
        call the class with x and y.
        The class returns the dielectric constant distribution in the cell.
    """
    eps_func = None
    def __init__(self, rel_r, cell_size_x, cell_size_y, eps_bulk, eps_hole=1.0):
        self.rel_r = rel_r
        self.r = self.rel_r*np.sqrt(cell_size_x*cell_size_y)
        self.r__2 = self.r**2
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
        self.half_cell_size_x = cell_size_x/2
        self.half_cell_size_y = cell_size_y/2
        self.eps_bulk = eps_bulk
        self.eps_hole = eps_hole
        @_periodically_continued(0, cell_size_x)
        def _x(x_):
            return x_
        @_periodically_continued(0, cell_size_y)
        def _y(y_):
            return y_
        self._x = _x
        self._y = _y
        self.eps_type = 'circle'
        self.FF = np.pi*self.rel_r**2
        self.avg_eps = self.eps_bulk*(1-self.FF) + self.eps_hole*self.FF

    def eps(self, x, y):
        x = self._x(x)
        y = self._y(y)
        return np.where((x - self.half_cell_size_x)**2 + (y - self.half_cell_size_y)**2 < self.r__2, self.eps_hole, self.eps_bulk)
    
    def __call__(self, x, y):
        return self.eps(x, y)