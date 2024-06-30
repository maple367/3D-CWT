import multiprocessing as mp
import uuid
import numpy as np
import os

class Data():
    def __init__(self) -> None:
        pass

    def load_npy(self, file_path:str):
        data = np.load(file_path)
        res = 1
        setattr(data, 'res', res)
