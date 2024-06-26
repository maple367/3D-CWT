import numpy as np
from typing import Callable
from .._boundary import _periodically_continued

# air hole class
class eps_circle():
    """
    The rectangular lattice with circular holes.
    The center of the circular hole is at the center of the cell.
    The cell corners are at (0, 0), (cell_size_x, 0), (0, cell_size_y), and (cell_size_x, cell_size_y).

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
    def __init__(self, rel_r, cell_size_x, cell_size_y, eps_bulk, eps_hole=1.0):
        self.rel_r = rel_r
        self.r = self.rel_r*np.sqrt(cell_size_x*cell_size_y)
        r__2 = self.r**2
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
        half_cell_size_x = cell_size_x/2
        half_cell_size_y = cell_size_y/2
        self.eps_bulk = eps_bulk
        self.eps_hole = eps_hole
        @_periodically_continued(0, cell_size_x)
        def _x(x_):
            return x_
        @_periodically_continued(0, cell_size_y)
        def _y(y_):
            return y_
        self._x = np.vectorize(_x)
        self._y = np.vectorize(_y)
        def eps(x_, y_): #TODO: Ctype call function to speedup
            x_ = self._x(x_)
            y_ = self._y(y_)
            if (x_ - half_cell_size_x)**2 + (y_ - half_cell_size_y)**2 < r__2:
                return eps_hole
            else:
                return eps_bulk
        self.eps = np.vectorize(eps)
    
    def __call__(self, x, y):
        return self.eps(x, y)
    
    def __getattr__(self, name: str):
        if name == 'eps_type':
            return 'circle'

    
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
        self._x = np.vectorize(_x)
        self._y = np.vectorize(_y)
        def eps(x_, y_):
            x_ = self._x(x_)
            y_ = self._y(y_)
            eps_val = self.eps_func(x_, y_)
            return eps_val
        self.eps = np.vectorize(eps)
    
    def __call__(self, x, y):
        return self.eps(x, y)
    
    def __getatt__(self, name: str):
        if name == 'eps_type':
            return 'userdefine'