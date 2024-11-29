# %%
import utils
import os
import pandas as pd
import numpy as np

dir_names = []
FFs = []
shapes = []
reses = []
for i in os.listdir('./history_res'):
    dir_names += [i]
    res = utils.Data(f'./history_res/{i}').load_all()
    FFs += [res[1]['materials'][0].__dict__['FF']]
    shapes += [res[1]['materials'][0].__dict__['eps_type']]
    reses += [res]
df = pd.DataFrame({'dir_name':dir_names, 'FF':FFs, 'shape':shapes, 'res':reses})

x = []
y = []
y1 = []
y3 = []
y4 = []
scatters = []

for i in df[df['shape']=='circle'].sort_values('FF')['res']:
    x += [i[1]['materials'][0].__dict__['FF']]
    y += [i[0]['delta']]
    y1 += [i[0]['alpha']]
    i_eigs_inf = np.argmin(np.real(i[0]['eigen_values']))
    y3 += [np.real(i[0]['eigen_values'][i_eigs_inf])]
    y4 += [np.imag(i[0]['eigen_values'][i_eigs_inf])]
    scatter = [[x[-1], _y] for _y in y[-1]]
    scatters += scatter

x = np.array(x) *100
y = np.array(y)
y1 = np.array(y1)
scatters = np.array(scatters)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.xlim([5, 25])
plt.show()

# %%
from scipy.interpolate import interp1d
def fix_line_data(y:np.ndarray, index0=None):
    x = np.arange(len(y))
    y_new = y[:2]
    index = np.array([np.arange(len(y[:2].T))]*len(y[:2]))
    for i in range(len(y)-2):
        if index0 is None:
            y_interp = interp1d(x[:2+i], y_new, kind='slinear', axis=0, fill_value='extrapolate')
            y_pred = y_interp(x[2+i])
            index_pred = np.argsort(y_pred)
            revers_index_pred = np.argsort(index_pred)
        else:
            revers_index_pred = index0[2+i]
        y_new = np.vstack([y_new, np.sort(y[2+i])[revers_index_pred]])
        index = np.vstack([index, revers_index_pred])
    return y_new, index

y_new, index0 = fix_line_data(y)
plt.plot(x, y_new)
plt.xlim([5, 25])
plt.show()

y1_new, index1 = fix_line_data(y1, index0)
plt.plot(x, y1_new)
plt.yscale('symlog', linthresh=1e-20)
plt.xlim([5, 25])
plt.show()
        
# %%
plt.plot(x, y1_new)
plt.plot(x, y4)
plt.yscale('symlog', linthresh=1e-20)
plt.xlim([5, 25])
plt.show()
# %%
plt.plot(x, y_new)
plt.plot(x, y3)
plt.xlim([5, 25])
plt.show()
# %%
