import numpy as np
from scipy.linalg import eig


class SGM():
    def __init__(self, C_mats, init_eig_guess:complex, size:float, resolution:int):
        self.C_mats = C_mats
        self.init_eig_guess = init_eig_guess
        self.size = size
        self.resolution = resolution

    def _mesh_grid_(self):
        self.dx = self.size/self.resolution
        self.dy = self.size/self.resolution
        self.c_grid = np.zeros((self.resolution, self.resolution), dtype=complex)
        
