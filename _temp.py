# %%
import numpy as np
import pandas as pd
import scipy.constants as const
import matplotlib.pyplot as plt
import utils


# %%
a = 0.298
df = pd.read_csv('FF.csv')
f_list = []
FF_list = []
for i in range(len(df)):
    FF = df['FF'][i]
    FF_list.append(FF)
    datas = utils.Data(f'./history_res/{df["uuid"][i]}').load_res()
    f_s = datas['omega']/(2*np.pi)
    f_s = f_s/(const.c*1e6/a)
    f_list.append(f_s)
# %%
f_array = np.ones((len(f_list), len(f_list[0])))
for i in range(len(f_list)):
    f_array[i] = f_list[i]
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(np.array(FF_list), f_array,)
ax.set_xlabel('FF')
ax.set_ylabel('f (c/a)')
plt.show()
# %%
