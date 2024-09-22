import numpy as np
from .sgm import SGM, SGM_2D
import os

class Data():
    def __init__(self, path:str):
        self.path = path

    def load_res(self):
        file_path = os.path.join(self.path,'CWT_res.npy')
        data = np.load(file_path, allow_pickle=True).item()
        return data

    def load_all(self):
        datas = []
        for _ in os.listdir(self.path):
            file_path = os.path.join(self.path,_)
            data = np.load(file_path, allow_pickle=True).item()
            datas.append(data)
        return datas