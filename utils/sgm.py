import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, inv
from scipy.sparse import csr_array, csc_array
import mph

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
                self.M[j:j+4, k:k+4] = np.eye(4)
                # for j+1, k
                # for j, k+1
                # no k+1 at the upper boundary
            elif ((origin+1)%(self.resolution+2)!=0) and ((origin+1)//(self.resolution+2)>=self.resolution+1): # out of right boundary
                # for j, k
                self.A[j:j+4, k:k+4] = np.eye(4)
                self.M[j:j+4, k:k+4] = np.eye(4)
                # for j+1, k
                # no j+1 at the right boundary
                # for j, k+1
            else: # out of upper right boundary
                # for j, k
                self.A[j:j+4, k:k+4] = np.eye(4)
                self.M[j:j+4, k:k+4] = np.eye(4)
                # for j+1, k
                # no j+1 at the right boundary
                # for j, k+1
                # no k+1 at the upper boundary
        # A @ x = w * M @ x now
        # because the M matrix is not complex hermitian and positive semi-definite, the eigs function will not work
        # turn A @ x = w * M @ x to M^-1 @ A @ x = w * x
        self.A = csr_array(self.A)
        self.M = csc_array(self.M)
        self.C = (inv(self.M)@self.A).toarray()
        # apply boundary condition to the matrix C
        for origin in range(len(XX)): # extent grid has no contribution to the matrix
            j = 4*origin
            # Rx(0,y) = 0
            if ((origin)//(self.resolution+2)<1):
                self.C[j, :] = 0.0
                self.C[:, j] = 0.0
            # Sx(L,y) = 0
            if ((origin)//(self.resolution+2)>=self.resolution+1):
                self.C[j+1, :] = 0.0
                self.C[:, j+1] = 0.0
            # Ry(x,0) = 0
            if ((origin)%(self.resolution+2)==0):
                self.C[j+2, :] = 0.0
                self.C[:, j+2] = 0.0
            # Sy(x,L) = 0
            if ((origin+1)%(self.resolution+2)==0):
                self.C[j+3, :] = 0.0
                self.C[:, j+3] = 0.0
        # print('Matrix size:', self.A.shape)
        # convert to csr format sparse matrix
        self.C = csr_array(self.C)


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
        self.Rx_c = (self.Rx[:,:-1]+self.Rx[:,1:])/2
        self.Sx_c = (self.Sx[:,:-1]+self.Sx[:,1:])/2
        self.Ry_c = (self.Ry[:-1,:]+self.Ry[1:,:])/2
        self.Sy_c = (self.Sy[:-1,:]+self.Sy[1:,:])/2
        self.P_dis = np.square(np.abs(self.Rx_c))+np.square(np.abs(self.Sx_c))+np.square(np.abs(self.Ry_c))+np.square(np.abs(self.Sy_c))
        self.P_stim = np.sum(self.P_dis)*np.square(self.size/self.resolution)*2*self._w_used_.imag
        self.P_edge = self.size/self.resolution*np.sum(np.square(np.abs(self.Rx[:,-1]))+np.square(np.abs(self.Sx[:,1]))+np.square(np.abs(self.Ry[-1,:]))+np.square(np.abs(self.Sy[1,:])))
        self.xi_rads = self.res['xi_rads']
        self.kappa_v = self.res['kappa_v']
        self.P_rad = 2*np.imag(self.kappa_v)*np.square(self.size/self.resolution)*np.sum(np.square(np.abs(self.xi_rads[0]*self.Rx_c+self.xi_rads[1]*self.Sx_c))+np.square(np.abs(self.xi_rads[2]*self.Ry_c+self.xi_rads[3]*self.Sy_c)))
        if show_plot:
            self._plot_sgm_()

    def _plot_sgm_(self):
        """
        plot the SGM mesh
        """
        plt.subplot(1,2,2)
        plt.imshow(np.abs(self.P_dis)/np.max(np.abs(self.P_dis)), cmap='hot')
        plt.colorbar()
        plt.title('P_dis norm')
        plt.show()
        plt.close()
        print('P_stim:', self.P_stim)
        print('P_edge:', self.P_edge)
        print('P_rad:', self.P_rad)
        print('P_total:', self.P_edge+self.P_rad)

class SGM_2D():
    '''
    Use 2D staggered grid method to solve 2D eigenvalue problem.

    Parameters
    ----------
    res : dict
        The result of the CWT solver.
    init_eig_guess : complex
        The initial guess of the eigenvalue.
    size : int
        The number of cells in the x or y direction.
    resolution : int
        The resolution of the mesh grid.

    Methods
    ----------

    '''
    def __init__(self, res:dict, init_eig_guess:complex, size:int, resolution:int):
        self.res = res
        self.C_mat_sum = self.res['C_mat_sum']
        self.init_eig_guess = init_eig_guess
        self.size = size
        self.resolution = resolution
        self.grad_i = 1j/(self.size*self.res['a']/self.resolution)
        self._construct_matrix_()

    def _construct_matrix_(self):
        if isinstance(self.resolution, int) == False:
            raise TypeError('resolution should be an integer.')
        num_gird = self.resolution+1
        self.size_eig_vec = (4*num_gird+2)*num_gird+num_gird*2
        self.C = np.zeros((self.size_eig_vec, self.size_eig_vec), dtype=complex)
        self.M = np.zeros((self.size_eig_vec, self.size_eig_vec), dtype=complex)
        for row in range(self.size_eig_vec):
            x_coor = row%(4*num_gird+2)
            y_coor = row//(4*num_gird+2)
            if x_coor<4*num_gird and y_coor<num_gird:
                self.M[row, row] = 1/2
                if x_coor%4<2:
                    self.M[row, row+4] = 1/2
                else:
                    if y_coor<num_gird-1:
                        self.M[row, row+(4*num_gird+2)] = 1/2
                    else:
                        self.M[row, row+(4*num_gird-x_coor//4*2)] = 1/2
            else:
                self.M[row, row] = 1
        for row in range(self.size_eig_vec):
            x_coor = row%(4*num_gird+2)
            y_coor = row//(4*num_gird+2)
            if x_coor%4==0 and x_coor<4*num_gird and y_coor<num_gird:
                self.C[row:row+4, row:row+4] = 1/2*self.C_mat_sum[:,:]
                self.C[row:row+4, row+4:row+6] = 1/2*self.C_mat_sum[:,:2]
                self.C[row,row] += -self.grad_i
                self.C[row+1,row+1] += self.grad_i
                self.C[row+2,row+2] += -self.grad_i
                self.C[row+3,row+3] += self.grad_i
                self.C[row,row+4] += self.grad_i
                self.C[row+1,row+5] += -self.grad_i
                if y_coor<num_gird-1:
                    self.C[row:row+4, row+(4*num_gird+4):row+(4*num_gird+6)] = 1/2*self.C_mat_sum[:,2:]
                    self.C[row+2,row+(4*num_gird+4)] += self.grad_i
                    self.C[row+3,row+(4*num_gird+5)] += -self.grad_i
                else:
                    self.C[row:row+4, row+(4*num_gird+2-x_coor//4*2):row+(4*num_gird+4-x_coor//4*2)] = 1/2*self.C_mat_sum[:,2:]
                    self.C[row+2,row+(4*num_gird+2-x_coor//4*2)] += self.grad_i
                    self.C[row+3,row+(4*num_gird+3-x_coor//4*2)] += -self.grad_i
            elif x_coor>=4*num_gird or y_coor>=num_gird:
                self.C[row, row] = 1
            else:
                pass
        A = csr_array(self.C)
        M = csc_array(self.M)
        self.A = inv(M)@A
        # boundary condition
        for row in range(self.size_eig_vec):
            x_coor = row%(4*num_gird+2)
            y_coor = row//(4*num_gird+2)
            if x_coor==0 and y_coor<num_gird:
                self.A[row, row] = 1e20
            if x_coor==4*num_gird+1 and y_coor<num_gird:
                self.A[row, row] = 1e20
            if y_coor==0 and x_coor%4==2:
                self.A[row, row] = 1e20
            if y_coor==num_gird and x_coor%2==1:
                self.A[row, row] = 1e20

    
    def run(self, w0=None, **kwargs):
        if w0 is None:
            w0 = self.init_eig_guess
        self.w0 = w0
        self.w = eigs(A=self.A, k=kwargs.get('k', 6), sigma=self.w0, which=kwargs.get('which', 'LM'))
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
        num_gird = self.resolution+1
        center_intensity = self.M@self._vec_raw_
        c_mesh = np.zeros((self.resolution+1, self.resolution+1), dtype=complex)
        self.R_x_c = c_mesh.copy()
        self.S_x_c = c_mesh.copy()
        self.R_y_c = c_mesh.copy()
        self.S_y_c = c_mesh.copy()
        for row in range(self.size_eig_vec):
            x_coor = row%(4*self.resolution+2)
            y_coor = row//(4*self.resolution+2)
            if x_coor<4*num_gird and y_coor<num_gird:
                if x_coor%4==0:
                    self.R_x_c[y_coor, x_coor//4] = center_intensity[row]
                    self.S_x_c[y_coor, x_coor//4] = center_intensity[row+1]
                    self.R_y_c[y_coor, x_coor//4] = center_intensity[row+2]
                    self.S_y_c[y_coor, x_coor//4] = center_intensity[row+3]
            else:
                pass




    

if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     res = {'C_mat_sum':np.array([[1, 1, 1, 1],
#                                     [1, 1, 1, 1],
#                                     [1, 1, 1, 1],
#                                     [1, 1, 1, 1]]),
#             'a':1.0,}
#     sgm = SGM_2D(res, 1.0j, 1, 1)

#     fig = plt.figure()
#     ax = fig.add_subplot(111)    
#     cm = ax.pcolormesh(sgm.M.real, shading='auto', edgecolors='black', linewidth=0.01)
#     ax.set_aspect('equal')
#     fig.colorbar(cm)
#     plt.show()

#     fig = plt.figure()
#     ax = fig.add_subplot(111)    
#     cm = ax.pcolormesh(sgm.C.real, shading='auto', edgecolors='black', linewidth=0.01)
#     ax.set_aspect('equal')
#     fig.colorbar(cm)
#     plt.show()
    pass