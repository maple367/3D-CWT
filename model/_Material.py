import numpy as np
import os
import pandas as pd


class material_class():
    """

    """
    def __init__(self) -> None:
        pass


base_dir = os.path.dirname(__file__)
__Al000GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.000GaAs.csv')
__Al097GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.097GaAs.csv')
__Al219GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.219GaAs.csv')
__Al342GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.342GaAs.csv')
__Al411GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.411GaAs.csv')
__Al452GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.452GaAs.csv')
__Al700GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.700GaAs.csv')

class AlxGaAs(material_class):
    """

    """
    def __init__(self, x:float, wavelength:float=0.98):
        self.x = x
        self.wavelength = wavelength
        self._cal_eps_()

    def _cal_eps_(self):
        if 0.000 <= self.x < 0.097:
            nk_b_data = pd.read_csv(__Al000GaAs_path__)
            nk_u_data = pd.read_csv(__Al097GaAs_path__)
            x_b = (0.097 - self.x)/(0.097-0.000)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.097 <= self.x < 0.219:
            nk_b_data = pd.read_csv(__Al097GaAs_path__)
            nk_u_data = pd.read_csv(__Al219GaAs_path__)
            x_b = (0.219 - self.x)/(0.219-0.097)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.219 <= self.x < 0.342:
            nk_b_data = pd.read_csv(__Al219GaAs_path__)
            nk_u_data = pd.read_csv(__Al342GaAs_path__)
            x_b = (0.342 - self.x)/(0.342-0.219)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.342 <= self.x < 0.411:
            nk_b_data = pd.read_csv(__Al342GaAs_path__)
            nk_u_data = pd.read_csv(__Al411GaAs_path__)
            x_b = (0.411 - self.x)/(0.411-0.342)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.411 <= self.x < 0.452:
            nk_b_data = pd.read_csv(__Al411GaAs_path__)
            nk_u_data = pd.read_csv(__Al452GaAs_path__)
            x_b = (0.452 - self.x)/(0.452-0.411)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        elif 0.452 <= self.x <= 0.700:
            nk_b_data = pd.read_csv(__Al452GaAs_path__)
            nk_u_data = pd.read_csv(__Al700GaAs_path__)
            x_b = (0.700 - self.x)/(0.700-0.452)
            n_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['n'])
            k_b = np.interp(self.wavelength, nk_b_data['wl'], nk_b_data['k'])
            n_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['n'])
            k_u = np.interp(self.wavelength, nk_u_data['wl'], nk_u_data['k'])
            self.n, self.k = n_b*x_b + n_u*(1-x_b), k_b*x_b + k_u*(1-x_b)
            self.epsilon = np.square(self.n + 1j*self.k)
        else:
            raise ValueError('x should be in [0.0000, 0.7000]')

class Air(material_class):
    """

    """
    def __init__(self):
        self.epsilon = 1.0 + 0.0j


class user_defined_material(material_class):
    """

    """
    def __init__(self, epsilon:complex):
        self.epsilon = epsilon


