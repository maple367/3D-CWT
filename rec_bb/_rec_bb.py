# XQ, Noel, 2024.7.2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(linewidth=np.inf)
import time

class data_recording():
    def __init__(self, scoring_package, x_name='none', y_name='none', recording_route='recording/'+time.strftime("%Y.%m.%d.%H.%M.%S.", time.localtime())):
        self.scoring_package = scoring_package
        self.x_name = x_name
        self.y_name = y_name
        self.recording_data = 'none'
        self.recording_route = recording_route
        self.recording_csv = self.recording_route+'/score_x_y.csv'
        self.recording_gra = self.recording_route+'/single_para_graphs'
        self.recording_others = self.recording_route+'/others'
        os.makedirs(self.recording_route)
        os.makedirs(self.recording_gra)
        os.makedirs(self.recording_others)
        self.recording_ite = 10
        self.ite = 0

    def bb(self, new_x):
        time_1 = time.time()
        score_y = self.scoring_package(new_x)
        time_2 = time.time() - time_1

        try:
            if self.recording_data == 'none':
                if self.x_name == 'none': self.x_name = ['x'+str(i) for i in range(len(new_x))]
                self.columns = ['score', 'y']+self.x_name+['time_used', 'year', 'month', 'day', 'hour', 'minute', 'second']
                self.recording_data = pd.DataFrame(columns = self.columns)
        except: pass

        self.recording_data.loc[self.recording_data.shape[0]] = score_y+list(new_x)+[time_2]+[i for i in time.localtime()[:-3]]
        if self.ite % self.recording_ite == 0:
            self.recording_data.to_csv(self.recording_csv, index=False)

            ite_ls = list(range(self.recording_data.shape[0]))
            for column in self.columns:
                plt.figure(figsize=(8,6))
                plt.plot(ite_ls, self.recording_data[column], color = 'b', linewidth=0.3)
                plt.xlabel('iteration', fontsize=15)
                plt.ylabel(column, fontsize=15)
                plt.grid()
                try: plt.savefig(self.recording_gra+'/'+column+'.png')
                except: pass
                plt.close()
        self.ite += 1

        return score_y[0]

