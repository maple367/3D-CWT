import numpy as np

class sgm_cal():
    """
    C_mat: 2D array, the C matrix, unit: 1/um
    size: the size of sovle region, unit: um
    resulution: int, the resulution of the solve region
    """

    def __init__(self, C_mat, eig_v, size=70, resulution=14):
        self.C_mat = C_mat
        self.eig_v = eig_v
        self.size = size
        self.resulution = resulution
        self.dx = size/resulution
        self.dy = self.dx
        self.__construct_sgm()
        self.abserr = 1
        self.relerr = 1

    def __construct_sgm(self):
        self.Rx = np.ones((self.resulution+2, self.resulution+2), dtype=np.complex128)*(1+1j)
        self.Sx = np.ones((self.resulution+2, self.resulution+2), dtype=np.complex128)*(1+1j)
        self.Ry = np.ones((self.resulution+2, self.resulution+2), dtype=np.complex128)*(1+1j)
        self.Sy = np.ones((self.resulution+2, self.resulution+2), dtype=np.complex128)*(1+1j)
        self.dRx = np.diff(self.Rx, axis=0)/self.dx
        self.dSx = np.diff(self.Sx, axis=0)/self.dx
        self.dRy = np.diff(self.Ry, axis=1)/self.dy
        self.dSy = np.diff(self.Sy, axis=1)/self.dy

    def __iter(self):
        """
        eig_v: the eigenvalue of the C matrix
        return: the value before iteration and after iteration
        """
        # apply the boundary condition
        self.Rx[0,:] = 0
        self.Sx[-1,:] = 0
        self.Ry[:,0] = 0
        self.Sy[:,-1] = 0
        __RS_mat__ = np.array([self.Rx, self.Sx, self.Ry, self.Sy])
        # update the value of dRx, dSx, dRy, dSy
        self.dRx = np.diff(self.Rx, axis=0)
        self.dSx = np.diff(self.Sx, axis=0)
        self.dRy = np.diff(self.Ry, axis=1)
        self.dSy = np.diff(self.Sy, axis=1)
        # apply the control equation
        for t in range(self.resulution+2):
            for j in range(self.resulution+1):
                for k in range(self.resulution+1):
                    Rx_c = self.Rx[j+1,k]+self.Rx[j,k]
                    Sx_c = self.Sx[j+1,k]+self.Sx[j,k]
                    Ry_c = self.Ry[j,k+1]+self.Ry[j,k]
                    Sy_c = self.Sy[j,k+1]+self.Sy[j,k]
                    RS_c_vec = np.array([[Rx_c], [Sx_c], [Ry_c], [Sy_c]])
                    dRx_c = self.dRx[j,k]
                    dSx_c = self.dSx[j,k]
                    dRy_c = self.dRy[j,k]
                    dSy_c = self.dSy[j,k]
                    dRS_c_vec = np.array([[dRx_c], [-dSx_c], [dRy_c], [-dSy_c]])
                    RS_c_vec = (0.5*(self.C_mat@RS_c_vec)+1j/(self.dx)*dRS_c_vec) *2/self.eig_v
                    # update the value of Rx, Sx, Ry, Sy
                    self.Rx[j+1,k] = RS_c_vec[0,0] - self.Rx[j,k]
                    self.Sx[j+1,k] = RS_c_vec[1,0] - self.Sx[j,k]
                    self.Ry[j,k+1] = RS_c_vec[2,0] - self.Ry[j,k]
                    self.Sy[j,k+1] = RS_c_vec[3,0] - self.Sy[j,k]

        self.abserr = np.abs(np.array([self.Rx, self.Sx, self.Ry, self.Sy])-__RS_mat__)
        self.relerr = np.max(np.abs(self.abserr[:,1:-1,1:-1]/__RS_mat__[:,1:-1,1:-1]))

    def runcal(self):
        i = 0
        while (self.relerr > 1e-8 and i < 1000):
            self.__iter()
            i += 1
        print(i)
        if i == 1000:
            print(f'The iteration is not converge. The relative error is {self.relerr}.')




if __name__ == '__main__':
    C = np.array([[ 0.00697052+8.57300964e-03j, -0.01027506+3.52204172e-03j,
                    0.0078502 +1.57819860e-07j, -0.00174034+4.37611050e-04j],
                  [-0.01811476-1.50834500e-02j,  0.00697052+5.39577832e-03j,
                   -0.00174034-4.37611050e-04j,  0.0078502 -1.57819860e-07j],
                  [ 0.0078502 -1.57819859e-07j, -0.00174034+4.37611048e-04j,
                    0.00696976+8.57484839e-03j, -0.01027432+3.51715450e-03j],
                  [-0.00174034-4.37611048e-04j,  0.0078502 +1.57819859e-07j,
                   -0.01811522-1.50824803e-02j,  0.00696976+5.39785450e-03j]])
    self = sgm_cal(C,5-3j)
    self.runcal()
    import matplotlib.pyplot as plt
    plt.imshow(np.abs(self.Rx)**2+np.abs(self.Sx)**2+np.abs(self.Ry)**2+np.abs(self.Sy)**2, cmap='hot')
    plt.colorbar()
    plt.show()
    # 3D plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(-self.dx, self.size+self.dx, self.dx)
    Y = np.arange(-self.dy, self.size+self.dy, self.dy)
    X, Y = np.meshgrid(X, Y)
    Z = np.abs(self.Rx)**2+np.abs(self.Sx)**2+np.abs(self.Ry)**2+np.abs(self.Sy)**2
    intensity = ax.plot_surface(X[1:-1,1:-1], Y[1:-1,1:-1], Z[1:-1,1:-1], cmap='hot')
    plt.show()
    