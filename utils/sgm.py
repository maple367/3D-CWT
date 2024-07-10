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
        mesh_grid = np.arange(self.resolution+2) # extent the mesh grid one more point to simplify the matrix construction
        XX, YY = np.meshgrid(mesh_grid, mesh_grid)
        XX = XX.flatten()
        YY = YY.flatten()
        self.A = np.zeros((len(YY)*4, len(XX)*4), dtype=np.longcomplex) # TODO: ensure the size
        self.M = self.A.copy()
        for origin in range(len(XX)): # extent grid has no contribution to the matrix
            if ((origin+1)%(self.resolution+2) !=0) and :
                j = 4*origin
                k = 4*origin
                # for j, k
                self.A[j:j+4, k:k+4] = self.C_mat_sum/2+self.grad_i
                self.M[j:j+4, k:k+4] = np.eye(4)/2
                # for j+1, k
                self.A[j:j+2, k+len(mesh_grid):k+len(mesh_grid)+4] = self.C_mat_sum[0:2,:]/2-self.grad_i[0:2,:]
                self.M[j:j+2, k+len(mesh_grid):k+len(mesh_grid)+4] = np.eye(4)[0:2,:]/2
                # for j, k+1
                self.A[j+2:j+4, k+4:k+8] = self.C_mat_sum[2:4,:]/2-self.grad_i[2:4,:]            
                self.M[j+2:j+4, k+4:k+8] = np.eye(4)[2:4,:]/2
                # if is extent grid
