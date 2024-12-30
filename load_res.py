# %%
import utils
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_set_file = 'mesh_data_set.csv'
df = pd.read_csv(data_set_file)
df.plot.scatter(x='Q', y='SE', marker='o', s=3)
plt.xscale('log')
plt.show()
# %%
df = df.dropna()
i = 0
datas = []
for line in df.values:
    para = utils.Data(f'./history_res/{line[2]}').load_para()
    res = utils.Data(f'./history_res/{line[2]}').load_model()
    eps_array = para['materials'][0].eps_distribtion_array
    datas.append({'eps_array': eps_array, 'Q': line[0], 'SE': line[1]})
    i += 1
    print(i)
np.save('mesh_data.npz', pd.DataFrame(datas).to_numpy())
# %%
