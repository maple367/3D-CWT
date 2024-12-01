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

for i in df[df['shape']=='RIT'].sort_values('FF')['res']:
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



# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(x, y)
axs[0, 0].set_title('delta before sort')
# axs[0, 0].set_xlim([5, 25])

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

y_new, index0 = fix_line_data(y)
axs[0, 1].plot(x, y_new)
axs[0, 1].set_title('delta after sort')
# axs[0, 1].set_xlim([5, 25])

y1_new, index1 = fix_line_data(y1, index0)
        
axs[1, 0].plot(x, y1_new, marker='o')
axs[1, 0].plot(x, y4, linestyle='--', label='min real')
axs[1, 0].set_title('alpha after sort')
# axs[1, 0].set_xlim([5, 25])
axs[1, 0].set_yscale('log')

axs[1, 1].plot(x, y_new, marker='o')
axs[1, 1].plot(x, y3, linestyle='--', label='min real')
axs[1, 1].set_title('delta after sort')
# axs[1, 1].set_xlim([5, 25])

plt.show()
# %%
