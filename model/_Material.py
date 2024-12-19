import numpy as np
import os
import pandas as pd


class material_class():
    """

    """
    def __init__(self):
        self.alpha = 4*np.pi*self.k/self.wavelength*1e4 #1/cm


base_dir = os.path.dirname(__file__)
# The following reference is from doi:10.1063/5.0039631
__Al000GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.000GaAs.csv')
__Al097GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.097GaAs.csv')
__Al219GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.219GaAs.csv')
__Al342GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.342GaAs.csv')
__Al411GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.411GaAs.csv')
__Al452GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.452GaAs.csv')
# The following reference is from doi:10.1063/1.343580
__Al700GaAs_path__ = os.path.join(base_dir, './mat_database/Al0.700GaAs.csv')
# The following reference is from doi:10.1021/nn501601e

class AlxGaAs(material_class):
    """
    Parameters
    ----------
    x : float
        The composition of AlxGaAs.
    wavelength : float
        The wavelength, unit in um.

    Returns
    -------
    out : class
        The class of AlxGaAs.
        self.epsilon : complex
            The epsilon of AlxGaAs.
        self.n : float
            The refractive index of AlxGaAs.
        self.k : float
            The extinction coefficient of AlxGaAs.
    """
    def __init__(self, x:float, wavelength:float=0.98):
        self.x = x
        self.wavelength = wavelength
        self._cal_eps_()
        super().__init__()

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
        

class AlxGaN(material_class):
    """
    Reference: doi:10.1063/1.366309

    Parameters
    ----------
    x : float
        The composition of AlxGaN.
    wavelength : float
        The wavelength, unit in um.

    Returns
    -------
    out : class
        The class of AlxGaN.
        self.epsilon : complex
            The epsilon of AlxGaN.
        self.n : float
            The refractive index of AlxGaN.
        self.k : float
            The extinction coefficient of AlxGaN.
    """
    Eg_GaN = 3.42 # eV
    Eg_AlN = 6.28 # eV
    b = 1.43 # eV
    def __init__(self, x:float, wavelength:float=0.45):
        self.x = x
        self.wavelength = wavelength
        self._cal_eps_()
        super().__init__()

    def _cal_eps_(self):
        E_lambda = 6.62607015e-34*299792458/(self.wavelength*1e-6)/1.60217662e-19
        Eg = (1-self.x)*self.Eg_GaN + self.x*self.Eg_AlN - self.x*(1-self.x)*self.b
        A = 3.17*np.sqrt(self.x)+9.98
        B = 2.66-2.2*self.x
        self.epsilon = A*np.square(Eg/E_lambda)*(2-np.sqrt(1+E_lambda/Eg)-np.sqrt(1-E_lambda/Eg)) + B
        self.n = np.real(np.sqrt(self.epsilon))
        self.k = np.imag(np.sqrt(self.epsilon))


class GaN(material_class):
    """
    Reference: doi:10.1063/1.365671

    Parameters
    ----------
    wavelength : float
        The wavelength, unit in um.

    Returns
    -------
    out : class
        The class of GaN.
        self.epsilon : complex
            The epsilon of GaN.
        self.n : float
            The refractive index of GaN.
        self.k : float
            The extinction coefficient of GaN.
    """
    E0  = 3.38    #eV
    G0  = 0.02    #eV
    A0  = 27      #eV**1.5
    A0x = 0.055   #eV**-1
    Γ0  = 0.06    #eV
    EminusG1A=6.8 #eV
    EminusG1B=7.9 #eV
    EminusG1C=9.0 #eV
    B1xA = 6.2    #eV
    B1xB = 0.6    #eV
    B1xC = 3.0    #eV
    Γ1A  = 0.78   #eV
    Γ1B  = 0.35   #eV
    Γ1C  = 1.0    #eV
    ε1   = 2.20
    def __init__(self, wavelength:float=0.45):
        self.wavelength = wavelength
        self.eV = 4.13566733e-1*2.99792458/self.wavelength
        self._cal_eps_()
        super().__init__()

    def _Epsilon_A_(self, eV):
        χ0 = (eV+1j*self.Γ0) / self.E0
        fχ0 = χ0**-2 * ( 2-(1+χ0)**0.5-(1-χ0)**0.5 )
        return self.A0*self.E0**-1.5 * fχ0
    
    def _Epsilon_Ax_(self, eV):
        y=0
        n = np.arange(1,1000)
        y = np.sum(self.A0x/n**3 / (self.E0-self.G0/n**2-eV-1j*self.Γ0))
        return y
    
    def _Epsilon_B_(self, eV):
        ε  = self.B1xA / (self.EminusG1A-eV-1j*self.Γ1A)
        ε += self.B1xB / (self.EminusG1B-eV-1j*self.Γ1B)
        ε += self.B1xC / (self.EminusG1C-eV-1j*self.Γ1C)
        return ε

    def _cal_eps_(self):
        self.epsilon = self._Epsilon_A_(self.eV) + self._Epsilon_Ax_(self.eV) + self._Epsilon_B_(self.eV) + self.ε1
        self.n = np.real(np.sqrt(self.epsilon))
        self.k = np.imag(np.sqrt(self.epsilon))

class InxGaN(material_class):
    """
    Reference: doi:10.1063/1.366309

    Parameters
    ----------
    x : float
        The composition of InxGaN.
    wavelength : float
        The wavelength, unit in um.

    Returns
    -------
    out : class
        The class of InxGaN.
        self.epsilon : complex
            The epsilon of InxGaN.
        self.n : float
            The refractive index of InxGaN.
        self.k : float
            The extinction coefficient of InxGaN.
    """
    Eg_GaN = 3.42 # eV
    Eg_InN = 0.81 # eV
    def __init__(self, x:float, wavelength:float=0.45):
        self.x = x
        self.wavelength = wavelength
        self._cal_eps_()
        super().__init__()

    def _cal_eps_(self):
        pass


def ITO(material_class):
    """

    """
    def __init__(self):
        self.epsilon = 1.0 + 0.0j


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


