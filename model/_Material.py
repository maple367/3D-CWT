import numpy as np
import os
import pandas as pd

base_dir = os.path.dirname(__file__)
__Al0000GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.0000GaAs.csv')
__Al0097GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.0097GaAs.csv')
__Al2190GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.2190GaAs.csv')
__Al3420GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.3420GaAs.csv')
__Al4110GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.4110GaAs.csv')
__Al4520GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.4520GaAs.csv')
__Al7000GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.7000GaAs.csv')

class AlxGaAs():
    """

    """
    def __init__(self, x:float, wavelength:float=0.98):
        self.x = x
        self.wavelength = wavelength
        self._cal_eps_()

    def _cal_eps_(self):
        if 0.0000 <= self.x < 0.0097:
            nk_b_data = pd.read_csv(__Al0000GaAs_path__)
            nk_u_data = pd.read_csv(__Al0097GaAs_path__)
            x_b = (0.0097 - self.x)/(0.0097-0.0000)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.0097 <= self.x < 0.2190:
            nk_b_data = pd.read_csv(__Al0097GaAs_path__)
            nk_u_data = pd.read_csv(__Al2190GaAs_path__)
            x_b = (0.2190 - self.x)/(0.2190-0.0097)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.2190 <= self.x < 0.3420:
            nk_b_data = pd.read_csv(__Al2190GaAs_path__)
            nk_u_data = pd.read_csv(__Al3420GaAs_path__)
            x_b = (0.3420 - self.x)/(0.3420-0.2190)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.3420 <= self.x < 0.4110:
            nk_b_data = pd.read_csv(__Al3420GaAs_path__)
            nk_u_data = pd.read_csv(__Al4110GaAs_path__)
            x_b = (0.4110 - self.x)/(0.4110-0.3420)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.4110 <= self.x < 0.4520:
            nk_b_data = pd.read_csv(__Al4110GaAs_path__)
            nk_u_data = pd.read_csv(__Al4520GaAs_path__)
            x_b = (0.4520 - self.x)/(0.4520-0.4110)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.4520 <= self.x <= 0.7000:
            nk_b_data = pd.read_csv(__Al4520GaAs_path__)
            nk_u_data = pd.read_csv(__Al7000GaAs_path__)
            x_b = (0.7000 - self.x)/(0.7000-0.4520)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
