import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

class SGM():
    def __init__(self, C_mat_sum, init_eig_guess:complex, size:float, resolution:int):
        self.C_mat_sum = C_mat_sum
        self.init_eig_guess = init_eig_guess
        self.size = size # um
        self.resolution = resolution
        self.err = 1.0
        self.tol = 1e-5
        self.grad_i = np.array([[-1.0j, 0.0, 0.0, 0.0],
                                [0.0, 1.0j, 0.0, 0.0],
                                [0.0, 0.0, -1.0j, 0.0],
                                [0.0, 0.0, 0.0, 1.0j]])/(self.size/self.resolution)
        self._construct_matrix_()

    def _construct_matrix_(self):
        """
        construct sparse matrix for the eigenvalue problem.
        eigenvalue problem: A @ x[i] = w[i] * M @ x[i]
        """
        if isinstance(self.resolution, int) == False:
            raise TypeError('resolution should be an integer.')
        mesh_grid = np.arange(self.resolution+1)
        XX, YY = np.meshgrid(mesh_grid, mesh_grid)
        XX = XX.flatten()
        YY = YY.flatten()
        self.A = np.zeros((self.resolution*self.resolution*4, self.resolution*self.resolution*4), dtype=np.longcomplex) # TODO: ensure the size
        self.M = self.A.copy()
        for origin in range(self.resolution+1):
            j = 4*origin
            k = 4*origin
            self.A[j:j+4, k:k+4] = self.C_mat_sum/2+self.grad_i
            self.A[j:j+2, k+4*self.resolution:k+4*self.resolution+4] = self.C_mat_sum[0:2,:]/2-self.grad_i[0:2,:]
            self.A[j+2:j+4, k+4:k+8] = self.C_mat_sum[2:4,:]/2-self.grad_i[2:4,:]
            self.M[j:j+4, k:k+4] = np.eye(4)/2
            self.M[j:j+2, k+4*self.resolution:k+4*self.resolution+4] = np.eye(4)[0:2,:]/2
            self.M[j+2:j+4, k+4:k+8] = np.eye(4)[2:4,:]/2
