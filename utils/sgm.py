import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, inv
from scipy.sparse import csr_array, csc_array

class SGM():
    def __init__(self, res:dict, init_eig_guess:complex, size:float, resolution:int):
        self.res = res
        self.C_mat_sum = self.res['C_mat_sum']
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
        because M is singular here, a pesudo-inverse M is used.
        """
        if isinstance(self.resolution, int) == False:
            raise TypeError('resolution should be an integer.')
        mesh_grid = np.arange(self.resolution+2) # extent the mesh grid one more point to simplify the matrix construction
        XX, YY = np.meshgrid(mesh_grid, mesh_grid)
        XX = XX.flatten()
        YY = YY.flatten()
        self.A = np.zeros((len(YY)*4, len(XX)*4), dtype=complex)
        self.M = np.zeros((len(YY)*4, len(XX)*4), dtype=complex)
        # self.pass_num = 0
        # self.pass_ls = []
        for origin in range(len(XX)): # extent grid has no contribution to the matrix
            j = 4*origin
            k = 4*origin
            if ((origin+1)%(self.resolution+2)!=0) and ((origin)//(self.resolution+2)<self.resolution+1): # in boundary
                # for j, k
                self.A[j:j+4, k:k+4] = self.C_mat_sum/2+self.grad_i
                self.M[j:j+4, k:k+4] = np.eye(4)/2
                # for j+1, k
                self.A[j:j+2, k+len(mesh_grid)*4:k+len(mesh_grid)*4+4] = self.C_mat_sum[0:2,:]/2-self.grad_i[0:2,:]
                self.M[j:j+2, k+len(mesh_grid)*4:k+len(mesh_grid)*4+4] = np.eye(4)[0:2,:]/2
                # for j, k+1
                self.A[j+2:j+4, k+4:k+8] = self.C_mat_sum[2:4,:]/2-self.grad_i[2:4,:]            
                self.M[j+2:j+4, k+4:k+8] = np.eye(4)[2:4,:]/2
                # self.pass_num += 1
                # self.pass_ls.append(origin)
            elif ((origin+1)%(self.resolution+2)==0) and ((origin+1)//(self.resolution+2)<self.resolution+1): # out of upper boundary
                # for j, k
                self.A[j:j+4, k:k+4] = np.eye(4)
                self.M[j:j+4, k:k+4] = np.eye(4)/2
                # for j+1, k
                self.M[j:j+2, k+len(mesh_grid)*4:k+len(mesh_grid)*4+4] = np.eye(4)[0:2,:]/2
                # for j, k+1
                # no k+1 at the upper boundary
            elif ((origin+1)%(self.resolution+2)!=0) and ((origin+1)//(self.resolution+2)>=self.resolution+1): # out of right boundary
                # for j, k
                self.A[j:j+4, k:k+4] = np.eye(4)
                self.M[j:j+4, k:k+4] = np.eye(4)/2
                # for j+1, k
                # no j+1 at the right boundary
                # for j, k+1
                self.M[j+2:j+4, k+4:k+8] = np.eye(4)[2:4,:]/2
            else: # out of upper right boundary
                # for j, k
                self.A[j:j+4, k:k+4] = np.eye(4)
                self.M[j:j+4, k:k+4] = np.eye(4)/2
                # for j+1, k
                # no j+1 at the right boundary
                # for j, k+1
                # no k+1 at the upper boundary
        # apply boundary condition
        for origin in range(len(XX)): # extent grid has no contribution to the matrix
            j = 4*origin
            # Rx(0,y) = 0
            if ((origin)//(self.resolution+2)<1):
                self.A[j, :] = 0.0
                self.A[:, j] = 0.0
            # Sx(L,y) = 0
            if ((origin)//(self.resolution+2)>=self.resolution+1):
                self.A[j+1, :] = 0.0
                self.A[:, j+1] = 0.0
            # Ry(x,0) = 0
            if ((origin)%(self.resolution+2)==0):
                self.A[j+2, :] = 0.0
                self.A[:, j+2] = 0.0
            # Sy(x,L) = 0
            if ((origin+1)%(self.resolution+2)==0):
                self.A[j+3, :] = 0.0
                self.A[:, j+3] = 0.0
        # print('Matrix size:', self.A.shape)
        self._A_raw_ = self.A.copy()
        self._M_raw_ = self.M.copy()
        # remove fixed boundary not contribute to the eigenvalue problem
        self.remove_index_ls = []
        for origin in range(self.A.shape[0]):
            if (self.A[origin, :]==0.0).all() and (self.A[:, origin]==0.0).all():
                self.remove_index_ls.append(origin)
        self.A = np.delete(self.A, self.remove_index_ls, axis=0)
        self.A = np.delete(self.A, self.remove_index_ls, axis=1)
        self.M = np.delete(self.M, self.remove_index_ls, axis=0)
        self.M = np.delete(self.M, self.remove_index_ls, axis=1)
        # print('Apply boundary matrix size:', self.A.shape)
        # convert to csr format sparse matrix
        # because the M matrix is not complex hermitian and positive semi-definite, the eigs function will not work
        # turn A @ x = w * M @ x to M^-1 @ A @ x = w * x
        self.A = csr_array(self.A)
        self.M = csc_array(self.M)
        self.C = inv(self.M)@self.A

    def run(self, w0=None, **kwargs):
        """
        run the SGM algorithm
        """
        if w0 is None:
            w0 = self.init_eig_guess
        self.w0 = w0
        # w'[i] = 1/(w[i]-sigma)
        self.w = eigs(A=self.C, k=kwargs.get('k', 6), sigma=self.w0, which=kwargs.get('which', 'LM') )
        if kwargs.get('show_plot', False):
            self._plot_eig_()
        return self.w
    
    def _plot_eig_(self):
        """
        plot the eigenvalues
        """
        plt.figure()
        plt.plot(self.w[0].real, self.w[0].imag, 'o')
        for i in range(len(self.w[0])):
            plt.text(self.w[0].real[i], self.w[0].imag[i], str(i))
        plt.plot(self.w0.real, self.w0.imag, '+', color='red', markersize=50)
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.show()
        plt.close()

    def construct_sgm_mesh(self, indices=0, show_plot=False):
        """
        construct the mesh for the SGM algorithm
        """
        self._w_used_ = self.w[0][indices]
        self._vec_raw_ = self.w[1][:,indices].copy()
        # reinstate the eigen vector with the removed rows and columns
        for index in sorted(self.remove_index_ls):
            self._vec_raw_ = np.insert(self._vec_raw_, index, 0.0)
        self.Rx = np.zeros((self.resolution+2, self.resolution+2), dtype=complex)
        self.Sx = np.zeros((self.resolution+2, self.resolution+2), dtype=complex)
        self.Ry = np.zeros((self.resolution+2, self.resolution+2), dtype=complex)
        self.Sy = np.zeros((self.resolution+2, self.resolution+2), dtype=complex)
        for i, value in enumerate(self._vec_raw_):
            origin = i//4
            column = origin//(self.resolution+2)
            row = origin%(self.resolution+2)
            if i%4==0:
                self.Rx[row, column] = value
            elif i%4==1:
                self.Sx[row, column] = value
            elif i%4==2:
                self.Ry[row, column] = value
            elif i%4==3:
                self.Sy[row, column] = value
        self.Rx = self.Rx[:-1,:]
        self.Sx = self.Sx[:-1,:]
        self.Ry = self.Ry[:,:-1]
        self.Sy = self.Sy[:,:-1]
        self.P_dis = np.square(np.abs((self.Rx[:,:-1]+self.Rx[:,1:])/2))+np.square(np.abs((self.Sx[:,:-1]+self.Sx[:,1:])/2))+np.square(np.abs((self.Ry[:-1,:]+self.Ry[1:,:])/2))+np.square(np.abs((self.Sy[:-1,:]+self.Sy[1:,:])/2))
        self.P_stim = np.sum(self.P_dis)*np.square(self.size/self.resolution)*2*self._w_used_.imag
        self.P_edge = self.size*np.sum(np.square(np.abs(self.Rx[:,-1]))+np.square(np.abs(self.Sx[:,1]))+np.square(np.abs(self.Ry[-1,:]))+np.square(np.abs(self.Sy[1,:])))
