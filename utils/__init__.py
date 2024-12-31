import numpy as np
from .sgm import SGM_2D
import os

class Data():
    def __init__(self, path:str):
        self.path = path
        self.loaded = False

    def load_model(self):
        if self.loaded: return print('Some data has been loaded.')
        file_path = os.path.join(self.path,'model.npy')
        data = np.load(file_path, allow_pickle=True).item()
        self.__dict__.update(data)
        self.loaded = True
        return data
    
    def load_para(self):
        import multiprocessing as mp
        if self.loaded: return print('Some data has been loaded.')
        file_path = os.path.join(self.path,'input_para.npy')
        data = np.load(file_path, allow_pickle=True).item()
        self.__dict__.update(data)
        self.loaded = True
        return data

    def get_all(self):
        datas = []
        for _ in os.listdir(self.path):
            file_path = os.path.join(self.path,_)
            print(file_path)
            data = np.load(file_path, allow_pickle=True).item()
            datas.append(data)
        return datas

def fix_line_data(y:np.ndarray, index0=None):
    '''
    Parameters
    ----------
    y : np.ndarray
        The data to be fixed. The shape of y is (m, n), m is the number of data points, n is the number of data.
    index0 : np.ndarray, optional
        The index of the data to be fixed. The default is None. The shape of index0 is (m, n).

    Returns
    -------
    y_new : np.ndarray
        The fixed data. The shape of y_new is (m, n).
    index : np.ndarray
        The index of the fixed data. The shape of index is (m, n).
    '''
    from scipy.interpolate import interp1d
    x = np.arange(len(y))
    if index0 is None:
        y_new = y[:2]
        index = np.array([np.arange(len(y[:2].T))]*len(y[:2]))
        # check line
        for i in range(len(y)-2):
            y_interp = interp1d(x[:2+i], y_new, kind='slinear', axis=0, fill_value='extrapolate')
            y_pred = y_interp(x[2+i])
            index_pred = np.argsort(y_pred)
            index_origin = np.argsort(y[2+i])
            revers_index_pred = np.argsort(index_pred)
            index_final = index_origin[revers_index_pred]
            y_new = np.vstack([y_new, y[2+i][index_final]])
            index = np.vstack([index, index_final])
        # check each one point
        for i in range(len(y)):
            y_new_without_i = np.delete(y_new, i, axis=0)
            y_interp = interp1d(x[np.arange(len(y))!=i], y_new_without_i, kind='slinear', axis=0, fill_value='extrapolate')
            y_pred = y_interp(x[i])
            index_pred = np.argsort(y_pred)
            index_origin = np.argsort(y[i])
            revers_index_pred = np.argsort(index_pred)
            index_final = index_origin[revers_index_pred]
            y_new[i] = y[i][index_final]
            index[i] = index_final
    else:
        y_new = np.empty_like(y)
        index = np.empty_like(index0)
        for i in range(len(y)):
            index_final = index0[i]
            y_new[i] = y[i][index_final]
            index[i] = index_final
    return y_new, index