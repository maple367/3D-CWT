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
        # apply boundary conditions
        self.Rx_grid[0,:] = -self.Rx_grid[1,:]
        self.Sx_grid[-1,:] = -self.Sx_grid[-2,:]
        self.Ry_grid[:,0] = -self.Ry_grid[:,1]
        self.Sy_grid[:,-1] = -self.Sy_grid[:,-2]
        # calculate d_grid
        dRx_grid_old = np.diff(self.Rx_grid, axis=0)
        dSx_grid_old = np.diff(self.Sx_grid, axis=0)
        dRy_grid_old = np.diff(self.Ry_grid, axis=1)
        dSy_grid_old = np.diff(self.Sy_grid, axis=1)
        # update grid
        for y in range(self.resolution+1):
            for x in range(self.resolution+1):
                Rx_c = (self.Rx_grid[x,y] + self.Rx_grid[x+1,y]) / 2
                Sx_c = (self.Sx_grid[x,y] + self.Sx_grid[x+1,y]) / 2
                Ry_c = (self.Ry_grid[x,y] + self.Ry_grid[x,y+1]) / 2
                Sy_c = (self.Sy_grid[x,y] + self.Sy_grid[x,y+1]) / 2
                RS_c_vec = np.array([[Rx_c], [Sx_c], [Ry_c], [Sy_c]])
                dRx = self.dRx_grid[x,y]
                dSx = self.dSx_grid[x,y]
                dRy = self.dRy_grid[x,y]
                dSy = self.dSy_grid[x,y]
                dRS_c_vec = np.array([[dRx], [-dSx], [dRy], [-dSy]])
                RS_c_vec = 2/(-1j*self.init_eig_guess)*(self.C_mat_sum@RS_c_vec + 1j/self.dx*dRS_c_vec)
                self.Rx_grid[x+1,y] = RS_c_vec[0,0] - self.Rx_grid[x,y]
                self.Sx_grid[x+1,y] = RS_c_vec[1,0] - self.Sx_grid[x,y]
                self.Ry_grid[x,y+1] = RS_c_vec[2,0] - self.Ry_grid[x,y]
                self.Sy_grid[x,y+1] = RS_c_vec[3,0] - self.Sy_grid[x,y]
        for y in range(self.resolution+1):
            for x in range(self.resolution+1):
                self.Rx_c_grid[x,y] = (self.Rx_grid[x,y] + self.Rx_grid[x+1,y]) / 2
                self.Sx_c_grid[x,y] = (self.Sx_grid[x,y] + self.Sx_grid[x+1,y]) / 2
                self.Ry_c_grid[x,y] = (self.Ry_grid[x,y] + self.Ry_grid[x,y+1]) / 2
                self.Sy_c_grid[x,y] = (self.Sy_grid[x,y] + self.Sy_grid[x,y+1]) / 2
                RS_c_vec = np.array([[self.Rx_c_grid[x,y]], [self.Sx_c_grid[x,y]], [self.Ry_c_grid[x,y]], [self.Sy_c_grid[x,y]]])
                dRS_c_vec = (self.init_eig_guess*RS_c_vec - self.C_mat_sum@RS_c_vec)/1j
                self.dRx_grid[x,y] = dRS_c_vec[0,0]
                self.dSx_grid[x,y] = -dRS_c_vec[1,0]
                self.dRy_grid[x,y] = dRS_c_vec[2,0]
                self.dSy_grid[x,y] = -dRS_c_vec[3,0]
        # calculate error
        self.err = np.max(np.abs(dRx_grid_old - self.dRx_grid)) + np.max(np.abs(dSx_grid_old - self.dSx_grid)) + np.max(np.abs(dRy_grid_old - self.dRy_grid)) + np.max(np.abs(dSy_grid_old - self.dSy_grid))
        print(f"\riter {self.iter}: max error {self.err}         ", end="", flush=True)

    def plot(self):
        self.fig, self.ax = plt.subplots()
        self.image = self.ax.imshow((np.abs(self.Rx_grid)**2), cmap='hot')
        self.ax.set_title('$|R_x|^2$')
        self.fig.colorbar(self.image)
        plt.show()