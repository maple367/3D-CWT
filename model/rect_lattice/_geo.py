import numpy as np
from .._boundary import _periodically_continued

# air hole class
class eps_circle:
    def __init__(self, r, cell_size_x, cell_size_y, eps_bulk, eps_hole=1.0):
        """
        The rectangular lattice with circular holes.
        The center of the circular hole is at the origin.

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
        """
        self.r__2 = r**2
        self.half_cell_size_x = cell_size_x/2
        self.half_cell_size_y = cell_size_y/2
        self.eps_bulk = eps_bulk
        self.eps_hole = eps_hole

def eps_circle(r, cell_size_x, cell_size_y, eps_bulk, eps_hole=1.0):
    """
    The rectangular lattice with circular holes.
    The center of the circular hole is at the origin.

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
    out : function
        The vectorized function of the dielectric constant distribution.
    """
    r__2 = r**2
    half_cell_size_x = cell_size_x/2
    half_cell_size_y = cell_size_y/2
    @_periodically_continued(0, cell_size_x)
    def _x(x_):
        return x_
    @_periodically_continued(0, cell_size_y)
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
    eps = np.vectorize(eps)
    return eps