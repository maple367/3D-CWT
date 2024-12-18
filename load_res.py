# %%
import utils
import os
import pandas as pd
import numpy as np

dir_names = []
FFs = []
shapes = []
reses = []
for i in os.listdir('./history_res')[:1]:
    dir_names += [i]
    res = utils.Data(f'./history_res/{i}').load_all()
    FFs += [res[1]['materials'][0].__dict__['FF']]
    shapes += [res[1]['materials'][0].__dict__['FF']]
    reses += [res]
df = pd.DataFrame({'dir_name':dir_names, 'FF':FFs, 'shape':shapes, 'res':reses})
# df.to_csv()
# %%
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

from utils import fix_line_data

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
