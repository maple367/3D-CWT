import numpy as np
import matplotlib.pyplot as plt


class SGM():
    def __init__(self, C_mat_sum, init_eig_guess:complex, size:float, resolution:int):
        self.C_mat_sum = C_mat_sum
        self.init_eig_guess = init_eig_guess
        self.size = size
        self.resolution = resolution
        self.err = 1.0
        self.tol = 1e-5
        self._mesh_grid_()

    def _mesh_grid_(self):
        self.dx = self.size/self.resolution
        self.dy = self.size/self.resolution
        c_grid = np.ones((self.resolution+2, self.resolution+2), dtype=np.complex128)
        x_grid = np.zeros((self.resolution+2, self.resolution+1), dtype=np.complex128)
        y_grid = np.zeros((self.resolution+1, self.resolution+2), dtype=np.complex128)
        d_grid = np.zeros((self.resolution+1, self.resolution+1), dtype=np.complex128)
        self.Rx_grid = x_grid.copy()
        self.Sx_grid = x_grid.copy()
        self.Ry_grid = y_grid.copy()
        self.Sy_grid = y_grid.copy()
        self.dRx_grid = d_grid.copy()
        self.dRy_grid = d_grid.copy()
        self.dSx_grid = d_grid.copy()
        self.dSy_grid = d_grid.copy()
        self.Rx_c_grid = d_grid.copy()
        self.Sx_c_grid = d_grid.copy()
        self.Ry_c_grid = d_grid.copy()
        self.Sy_c_grid = d_grid.copy()

    def run(self, max_iter=1000):
        self.max_iter = max_iter
        self.iter = 0
        self._initializor_()
        while (self.err > self.tol) and (self.iter <= self.max_iter):
            self._update_()
            self.iter += 1

    def _initializor_(self):
        self.Rx_grid[1,1] = 1+0j
        self.Sx_grid[1,1] = 0+0j

    def _update_(self):
        # calculate d_grid
        self.dRx_grid = self.Rx_grid[1:,:] - self.Rx_grid[:-1,:]
        self.dSx_grid = self.Sx_grid[1:,:] - self.Sx_grid[:-1,:]
        self.dRy_grid = self.Ry_grid[:,1:] - self.Ry_grid[:,:-1]
        self.dSy_grid = self.Sy_grid[:,1:] - self.Sy_grid[:,:-1]
        # save old values
        dRx_grid_old = self.dRx_grid.copy()
        dSx_grid_old = self.dSx_grid.copy()
        dRy_grid_old = self.dRy_grid.copy()
        dSy_grid_old = self.dSy_grid.copy()
        # apply boundary conditions
        self.Rx_grid[0,:] = -self.Rx_grid[1,:]
        self.Sx_grid[-1,:] = -self.Sx_grid[-2,:]
        self.Ry_grid[:,0] = -self.Ry_grid[:,1]
        self.Sy_grid[:,-1] = -self.Sy_grid[:,-2]
        # update grid
        for y in range(self.resolution+1):
            for x in range(self.resolution+1):
                self.Rx_grid[x+1,y] = 2/(-1j*self.init_eig_guess)*(self.dRx_grid[x,y]/self.dx - 1j/2*(self.C_mat_sum[0,0]*(self.Rx_grid[x,y]+self.Rx_grid[x+1,y]) + self.C_mat_sum[0,1]*(self.Sx_grid[x,y]+self.Sx_grid[x+1,y])+self.C_mat_sum[0,2]*(self.Ry_grid[x,y]+self.Ry_grid[x,y+1])+self.C_mat_sum[0,3]*(self.Sy_grid[x,y]+self.Sy_grid[x,y+1]))) - self.Rx_grid[x,y]
                self.Sx_grid[x+1,y] = 2/(-1j*self.init_eig_guess)*(-self.dSx_grid[x,y]/self.dx - 1j/2*(self.C_mat_sum[1,0]*(self.Rx_grid[x,y]+self.Rx_grid[x+1,y]) + self.C_mat_sum[1,1]*(self.Sx_grid[x,y]+self.Sx_grid[x+1,y])+self.C_mat_sum[1,2]*(self.Ry_grid[x,y]+self.Ry_grid[x,y+1])+self.C_mat_sum[1,3]*(self.Sy_grid[x,y]+self.Sy_grid[x,y+1]))) - self.Sx_grid[x,y]
                self.Ry_grid[x,y+1] = 2/(-1j*self.init_eig_guess)*(self.dRy_grid[x,y]/self.dy - 1j/2*(self.C_mat_sum[2,0]*(self.Rx_grid[x,y]+self.Rx_grid[x+1,y]) + self.C_mat_sum[2,1]*(self.Sx_grid[x,y]+self.Sx_grid[x+1,y])+self.C_mat_sum[2,2]*(self.Ry_grid[x,y]+self.Ry_grid[x,y+1])+self.C_mat_sum[2,3]*(self.Sy_grid[x,y]+self.Sy_grid[x,y+1]))) - self.Ry_grid[x,y]
                self.Sy_grid[x,y+1] = 2/(-1j*self.init_eig_guess)*(-self.dSy_grid[x,y]/self.dy - 1j/2*(self.C_mat_sum[3,0]*(self.Rx_grid[x,y]+self.Rx_grid[x+1,y]) + self.C_mat_sum[3,1]*(self.Sx_grid[x,y]+self.Sx_grid[x+1,y])+self.C_mat_sum[3,2]*(self.Ry_grid[x,y]+self.Ry_grid[x,y+1])+self.C_mat_sum[3,3]*(self.Sy_grid[x,y]+self.Sy_grid[x,y+1]))) - self.Sy_grid[x,y]
        self.dRx_grid = self.Rx_grid[1:,:] - self.Rx_grid[:-1,:]
        self.dSx_grid = self.Sx_grid[1:,:] - self.Sx_grid[:-1,:]
        self.dRy_grid = self.Ry_grid[:,1:] - self.Ry_grid[:,:-1]
        self.dSy_grid = self.Sy_grid[:,1:] - self.Sy_grid[:,:-1]
        # calculate error
        self.err = np.max(np.abs(dRx_grid_old - self.dRx_grid)) + np.max(np.abs(dSx_grid_old - self.dSx_grid)) + np.max(np.abs(dRy_grid_old - self.dRy_grid)) + np.max(np.abs(dSy_grid_old - self.dSy_grid))
        print(f"\riter {self.iter}: max error {self.err}         ", end="", flush=True)

    def plot(self):
        self.fig, self.ax = plt.subplots()
        self.image = self.ax.imshow((np.abs(self.Rx_grid)**2), cmap='hot')
        self.ax.set_title('$|R_x|^2$')
        self.fig.colorbar(self.image)
        plt.show()