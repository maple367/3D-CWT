# %%
import numpy as np
from model import AlxGaAs
from model.rect_lattice import eps_mesh
from calculator import xi_calculator_DFT
import multiprocessing as mp

if __name__ == '__main__':
    GaAs_eps = AlxGaAs(0).epsilon
    # fix seed
    np.random.seed(2333)

    # %%
    i_iter = 0
    while i_iter <= 100:
        eps_sample = np.random.random_sample((32, 32))
        FF = 0.28
        eps_thresh = np.percentile(eps_sample, FF*100)
        eps_array = np.where(eps_sample>eps_thresh, GaAs_eps.real, 1.0)
        eps_dis = eps_mesh(eps_array)
        i_iter += 1
    # %%
    lock = mp.Manager().Lock()
    dft_res = xi_calculator_DFT(eps_dis,pathname_suffix='dft_test',lock=lock)
    idft = np.fft.ifft2(dft_res.xi_array)*dft_res.resolution**2

    # %%
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2)
    im1 = axs[0, 0].imshow(eps_array)
    axs[0, 0].set_title('eps_array')
    im2 = axs[0, 1].imshow(np.abs(idft))
    axs[0, 1].set_title('ifft_xi_array')
    im3 = axs[1, 0].imshow(np.real(dft_res.xi_array))
    axs[1, 0].set_title('Re(xi_array)')
    im4 = axs[1, 1].imshow(np.imag(dft_res.xi_array))
    axs[1, 1].set_title('Im(xi_array)')
    for i, ax in enumerate(axs.flat):
        ax.set_axis_off()
        cb = fig.colorbar([im1, im2, im3, im4][i], ax=ax)

    plt.show()

    # %%
    from utils import Data
    data = Data('./history_res/20241122171611_17b81992da03478da817813b321d113c').load_all()
    xi_array_calculator = data[6]
    xi_00 = data[1]['epsilons'][0].avg_eps
    xi_array = np.empty((23, 23), dtype=complex)
    for i in range(-11,12):
        for j in range(-11,12):
            if i == 0 and j == 0:
                xi_array[i,j] = xi_00
            else:
                xi_array[i,j] = xi_array_calculator[i,j]
    idft = np.fft.ifft2(xi_array)*xi_array.shape[0]*xi_array.shape[1]
    plt.imshow(np.imag(idft))
    plt.colorbar()
    plt.show()
    # %%
