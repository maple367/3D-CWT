import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, inv
from scipy.sparse import csr_array, csc_array


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
        # boundary condition
        for row in range(self.size_eig_vec):
            x_coor = row%(4*num_gird+2)
            y_coor = row//(4*num_gird+2)
            if x_coor==0 and y_coor<num_gird:
                self.C[row, row] = 1e8
            if x_coor==4*num_gird+1 and y_coor<num_gird:
                self.C[row, row] = 1e8
            if y_coor==0 and x_coor%4==2:
                self.C[row, row] = 1e8
            if y_coor==num_gird and x_coor%2==1:
                self.C[row, row] = 1e8

    
    def run(self, w0=None, **kwargs):
        M = csc_array(self.M)
        self.A = inv(M)@csr_array(self.C)
        if w0 is None:
            w0 = self.init_eig_guess
        self.w0 = w0
        self.w = eigs(A=self.A, k=kwargs.get('k', 6),sigma=self.w0, which=kwargs.get('which', 'LM'))
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
    import matplotlib.pyplot as plt
    res = {'C_mat_sum':np.array([[1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 1]]),
            'a':1.0,}
    sgm = SGM_2D(res, 1.0j, 1, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)    
    cm = ax.pcolormesh(sgm.M.real, shading='auto', edgecolors='black', linewidth=0.01)
    ax.set_aspect('equal')
    fig.colorbar(cm)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)    
    cm = ax.pcolormesh(sgm.C.real, shading='auto', edgecolors='black', linewidth=0.01)
    ax.set_aspect('equal')
    fig.colorbar(cm)
    plt.show()
    pass